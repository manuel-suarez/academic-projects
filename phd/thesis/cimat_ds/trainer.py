import os
import time
import torch
import logging
import pandas as pd

from calculate_outcomes import outcomes, calculate_outcomes
from calculate_metrics import metrics, calculate_metrics, calculate_roc_metrics

save_every_epochs = 20

# Define ROC and AUC thresholds
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def initialize_metrics(
    initial_metrics, steps, roc_metrics=False, thresholds=thresholds
):
    for step in steps:
        # Loss
        step_loss = {f"{step}_loss": []}
        initial_metrics.update(step_loss)
        # Outcomes
        step_outcomes = {f"{step}_{outcome}": [] for outcome in outcomes}
        initial_metrics.update(step_outcomes)
        # Metrics
        step_metrics = {f"{step}_{metric}": [] for metric in metrics}
        initial_metrics.update(step_metrics)
        # Training time
        step_time = {f"{step}_time": []}
        initial_metrics.update(step_time)
        # ROC and AUC (only for valid and test steps)
        if roc_metrics:
            threshold_tpr_metrics = {
                f"{step}_{threshold}_tpr": [] for threshold in thresholds
            }
            initial_metrics.update(threshold_tpr_metrics)
            threshold_fpr_metrics = {
                f"{step}_{threshold}_fpr": [] for threshold in thresholds
            }
            initial_metrics.update(threshold_fpr_metrics)
            auc_metric = {f"{step}_auc": []}
            initial_metrics.update(auc_metric)
    return initial_metrics


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        epochs,
        device,
        metrics_path,
        weights_path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.metrics_path = metrics_path
        self.weights_path = weights_path

    def train_step(
        self,
        dataloader,
        epoch,
        batch_metrics,
        outcomes=outcomes,
        metrics=metrics,
        thresholds=thresholds,
    ):
        train_loss = 0.0
        epoch_outcomes = {f"{outcome}": 0 for outcome in outcomes}
        roc_outcomes = {
            threshold: {f"{outcome}": 0 for outcome in outcomes}
            for threshold in thresholds
        }
        self.model.train()
        for index, (image, label) in enumerate(dataloader):
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            # Get predictions
            output = self.model(image)
            # Calculate loss
            loss = self.loss_fn(output, label)
            # Get prediction outcomes
            train_outcomes = calculate_outcomes(
                output, label, use_sigmoid=True, use_threshold=True
            )
            # Get ROC/AUC outcomes calculation
            for threshold in thresholds:
                # Get threshold outcomes
                threshold_outcomes = calculate_outcomes(
                    output,
                    label,
                    use_sigmoid=True,
                    use_threshold=True,
                    p_threshold=threshold,
                )
                # Accumulate
                for outcome in outcomes:
                    roc_outcomes[threshold][outcome] += threshold_outcomes[outcome]

            # Get prediction metrics
            train_metrics = calculate_metrics(train_outcomes)
            # Update training
            loss.backward()
            self.optimizer.step()
            # Register batch loss, outcomes and metrics
            batch_metrics["epoch"].append(epoch)
            batch_metrics["batch"].append(index + 1)
            batch_metrics["train_loss"].append(loss.item())
            batch_metrics["start_time"].append("")
            batch_metrics["train_time"].append("")
            for outcome in outcomes:
                batch_metrics[f"train_{outcome}"].append(train_outcomes[outcome])
                # Accumulate epoch outcomes for epoch metrics
                epoch_outcomes[outcome] += train_outcomes[outcome]
            for metric in metrics:
                batch_metrics[f"train_{metric}"].append(train_metrics[metric])

            train_loss += loss.item()
        # Calculate epoch loss (mean of loss batches)
        train_loss = train_loss / len(dataloader)
        # Calculate metrics with epoch outcomes (the sum of batch outcomes instead of the mean of batch metrics)
        train_metrics = calculate_metrics(epoch_outcomes)
        # Accumulate values
        tpr_values = []
        fpr_values = []
        # Calculate ROC
        tpr_values.append(0.0)
        fpr_values.append(0.0)
        for threshold in thresholds:
            threshold_metrics = calculate_roc_metrics(roc_outcomes[threshold])
            threshold_tpr = threshold_metrics["tpr"]
            threshold_fpr = threshold_metrics["fpr"]
            # Store in metrics
            train_metrics[f"{threshold}_tpr"] = threshold_tpr
            train_metrics[f"{threshold}_fpr"] = threshold_fpr
            # Accumulcate TPR/FPR for AUC calculation
            tpr_values.append(threshold_tpr)
            fpr_values.append(threshold_fpr)
        # Calculate AUC
        auc_value = 0.0
        for i in range(1, len(fpr_values)):
            auc_value += (
                (fpr_values[i] - fpr_values[i - 1])
                * (tpr_values[i] + tpr_values[i - 1])
                / 2
            )
        train_metrics["auc"] = auc_value
        return {
            "loss": train_loss,
            "outcomes": epoch_outcomes,
            "metrics": train_metrics,
            "time": time.time(),
        }

    def valid_step(
        self,
        dataloader,
        epoch,
        batch_metrics,
        outcomes=outcomes,
        metric=metrics,
        thresholds=thresholds,
    ):
        valid_loss = 0.0
        epoch_outcomes = {f"{outcome}": 0 for outcome in outcomes}
        roc_outcomes = {
            threshold: {f"{outcome}": 0 for outcome in outcomes}
            for threshold in thresholds
        }
        self.model.eval()
        for index, (image, label) in enumerate(dataloader):
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                # Get predictions
                output = self.model(image)
                # Calculate loss
                loss = self.loss_fn(output, label)
                # Get prediction outcomes
                valid_outcomes = calculate_outcomes(
                    output, label, use_sigmoid=True, use_threshold=True
                )
                # Get ROC/AUC outcomes calculation
                for threshold in thresholds:
                    # Get threshold outcomes
                    threshold_outcomes = calculate_outcomes(
                        output,
                        label,
                        use_sigmoid=True,
                        use_threshold=True,
                        p_threshold=threshold,
                    )
                    # Accumulate
                    for outcome in outcomes:
                        roc_outcomes[threshold][outcome] += threshold_outcomes[outcome]

                # Get prediction metrics
                valid_metrics = calculate_metrics(valid_outcomes)
                # Register batch loss, outcomes and metrics
                batch_metrics["epoch"].append(epoch)
                batch_metrics["batch"].append(index + 1)
                batch_metrics["valid_loss"].append(loss.item())
                batch_metrics["start_time"].append("")
                batch_metrics["valid_time"].append("")
                for outcome in outcomes:
                    batch_metrics[f"valid_{outcome}"].append(valid_outcomes[outcome])
                    # Accumulate epoch outcomes for epoch metrics
                    epoch_outcomes[outcome] += valid_outcomes[outcome]
                for metric in metrics:
                    batch_metrics[f"valid_{metric}"].append(valid_metrics[metric])

                valid_loss += loss.item()
        # Calculate epoch loss (mean of loss batches)
        valid_loss = valid_loss / len(dataloader)
        # Calculate metrics with epoch outcomes (the sum of outcomes instead of the mean of batch metrics)
        valid_metrics = calculate_metrics(epoch_outcomes)
        # Accumulate values
        tpr_values = []
        fpr_values = []
        # Calculate ROC
        tpr_values.append(0.0)
        fpr_values.append(0.0)
        for threshold in thresholds:
            threshold_metrics = calculate_roc_metrics(roc_outcomes[threshold])
            threshold_tpr = threshold_metrics["tpr"]
            threshold_fpr = threshold_metrics["fpr"]
            # Store in metrics
            valid_metrics[f"{threshold}_tpr"] = threshold_tpr
            valid_metrics[f"{threshold}_fpr"] = threshold_fpr
            # Accumulcate TPR/FPR for AUC calculation
            tpr_values.append(threshold_tpr)
            fpr_values.append(threshold_fpr)
        # Calculate AUC
        auc_value = 0.0
        for i in range(1, len(fpr_values)):
            auc_value += (
                (fpr_values[i] - fpr_values[i - 1])
                * (tpr_values[i] + tpr_values[i - 1])
                / 2
            )
        valid_metrics["auc"] = auc_value
        return {
            "loss": valid_loss,
            "outcomes": epoch_outcomes,
            "metrics": valid_metrics,
            "time": time.time(),
        }

    def test_step(
        self,
        dataloader,
        epoch,
        batch_metrics,
        outcomes=outcomes,
        metric=metrics,
        thresholds=thresholds,
    ):
        test_loss = 0.0
        epoch_outcomes = {f"{outcome}": 0 for outcome in outcomes}
        roc_outcomes = {
            threshold: {f"{outcome}": 0 for outcome in outcomes}
            for threshold in thresholds
        }
        self.model.eval()
        for index, (image, label) in enumerate(dataloader):
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                # Get predictions
                output = self.model(image)
                # Calculate loss
                loss = self.loss_fn(output, label)
                # Get prediction outcomes
                test_outcomes = calculate_outcomes(
                    output, label, use_sigmoid=True, use_threshold=True
                )
                # Get ROC/AUC outcomes calculation
                for threshold in thresholds:
                    # Get threshold outcomes
                    threshold_outcomes = calculate_outcomes(
                        output,
                        label,
                        use_sigmoid=True,
                        use_threshold=True,
                        p_threshold=threshold,
                    )
                    # Accumulate
                    for outcome in outcomes:
                        roc_outcomes[threshold][outcome] += threshold_outcomes[outcome]

                # Get prediction metrics
                test_metrics = calculate_metrics(test_outcomes)
                # Register batch loss, outcomes and metrics
                batch_metrics["epoch"].append(epoch)
                batch_metrics["batch"].append(index + 1)
                batch_metrics["test_loss"].append(loss.item())
                batch_metrics["start_time"].append("")
                batch_metrics["test_time"].append("")
                for outcome in outcomes:
                    batch_metrics[f"test_{outcome}"].append(test_outcomes[outcome])
                    # Accumulate epoch outcomes for epoch metrics
                    epoch_outcomes[outcome] += test_outcomes[outcome]
                for metric in metrics:
                    batch_metrics[f"test_{metric}"].append(test_metrics[metric])

                test_loss += loss.item()
        # Calculate epoch loss (mean of loss batches)
        test_loss = test_loss / len(dataloader)
        # Calculate metrics with epoch outcomes (the sum of outcomes instead of the mean of batch metrics)
        test_metrics = calculate_metrics(epoch_outcomes)
        # Accumulate values
        tpr_values = []
        fpr_values = []
        # Calculate ROC
        tpr_values.append(0.0)
        fpr_values.append(0.0)
        for threshold in thresholds:
            threshold_metrics = calculate_roc_metrics(roc_outcomes[threshold])
            threshold_tpr = threshold_metrics["tpr"]
            threshold_fpr = threshold_metrics["fpr"]
            # Store in metrics
            test_metrics[f"{threshold}_tpr"] = threshold_tpr
            test_metrics[f"{threshold}_fpr"] = threshold_fpr
            # Accumulcate TPR/FPR for AUC calculation
            tpr_values.append(threshold_tpr)
            fpr_values.append(threshold_fpr)
        # Calculate AUC
        auc_value = 0.0
        for i in range(1, len(fpr_values)):
            auc_value += (
                (fpr_values[i] - fpr_values[i - 1])
                * (tpr_values[i] + tpr_values[i - 1])
                / 2
            )
        test_metrics["auc"] = auc_value
        return {
            "loss": test_loss,
            "outcomes": epoch_outcomes,
            "metrics": test_metrics,
            "time": time.time(),
        }

    # Fake step to get empty values for test step
    def zeros_step(self, outcomes=outcomes, metrics=metrics):
        zeros_metrics = {f"{metric}": "" for metric in metrics}
        for threshold in thresholds:
            zeros_metrics[f"{threshold}_tpr"] = ""
            zeros_metrics[f"{threshold}_fpr"] = ""
        zeros_metrics["auc"] = ""
        return {
            "loss": "",
            "outcomes": {f"{outcome}": "" for outcome in outcomes},
            "metrics": zeros_metrics,
            "time": "",
        }

    def fit(
        self,
        dataloaders,
        outcomes=outcomes,
        metrics=metrics,
        thresholds=thresholds,
    ):
        train_dataloader, valid_dataloader, test_dataloader = dataloaders
        # Due to nonlinearity operations on metrics calculation we need to calculate them
        # at several levels of dissagregation:
        # - Metrics per batch (at each evaluated batch)
        # - Metrics per epoch (at each epoch of the step)
        # - Metrics per step (at each step of the training process (train,valid,test))
        #   The train and valid metrics will be calculated on each level, the test metrics must be calculated
        #     at the end of the training process (theoretically we must adjust the hyperparameters of the
        #     training process using the differences between the train and valid metrics however in this
        #     moment we are not doing that) however we will calculate them on 20,40,60,80 and 100 epochs to
        #     simulate different ending training steps

        # Initialize dictionary for batch metrics
        batch_train_metrics = initialize_metrics(
            {"epoch": [], "batch": [], "start_time": []}, ["train"], roc_metrics=False
        )
        batch_valid_metrics = initialize_metrics(
            {"epoch": [], "batch": [], "start_time": []}, ["valid"], roc_metrics=False
        )
        batch_test_metrics = initialize_metrics(
            {"epoch": [], "batch": [], "start_time": []}, ["test"], roc_metrics=False
        )
        # Initialize dictionary for epoch metrics
        epoch_metrics = initialize_metrics(
            {"epoch": [], "start_time": []},
            ["train", "valid", "test"],
            roc_metrics=True,
        )
        # Initialize dictionary for training process metrics
        training_metrics = initialize_metrics(
            {"start_time": []},
            ["train", "valid", "test"],
        )
        training_losses = {"train": 0.0, "valid": 0.0, "test": 0.0}
        # Initialize global outcomes
        training_outcomes = {
            f"{step}": {f"{outcome}": 0 for outcome in outcomes}
            for step in ["train", "valid", "test"]
        }

        logging.info("--== Training start ==--")
        print("--== Training start ==--")
        startTime = time.time()

        training_metrics["start_time"].append(time.time())
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_outputs = self.train_step(
                train_dataloader,
                epoch,
                batch_train_metrics,
            )
            valid_outputs = self.valid_step(
                valid_dataloader,
                epoch,
                batch_valid_metrics,
            )
            # Run test step each 20 epochs
            if (epoch + 1) % save_every_epochs == 0:
                # Perform test step
                test_outputs = self.test_step(
                    test_dataloader,
                    epoch,
                    batch_test_metrics,
                )
            else:
                test_outputs = self.zeros_step()

            epoch_outputs = {
                "train": train_outputs,
                "valid": valid_outputs,
                "test": test_outputs,
            }
            epoch_metrics["epoch"].append(epoch + 1)
            epoch_metrics["start_time"].append(epoch_start_time)
            # Register epoch loss, outcomes, and metrics
            for step in ["train", "valid", "test"]:
                # Register loss
                epoch_metrics[f"{step}_loss"].append(epoch_outputs[step]["loss"])
                training_losses[f"{step}"] += (
                    0
                    if step == "test" and (epoch + 1) % save_every_epochs != 0
                    else epoch_outputs[step]["loss"]
                )
                # Register outcomes
                for outcome in outcomes:
                    epoch_metrics[f"{step}_{outcome}"].append(
                        epoch_outputs[step]["outcomes"][outcome]
                    )
                # Register metrics
                for metric in metrics:
                    epoch_metrics[f"{step}_{metric}"].append(
                        epoch_outputs[step]["metrics"][metric]
                    )
                # Register ROC per threshold
                for threshold in thresholds:
                    epoch_metrics[f"{step}_{threshold}_tpr"].append(
                        epoch_outputs[step]["metrics"][f"{threshold}_tpr"]
                    )
                    epoch_metrics[f"{step}_{threshold}_fpr"].append(
                        epoch_outputs[step]["metrics"][f"{threshold}_fpr"]
                    )
                # Register AUC
                epoch_metrics[f"{step}_auc"].append(
                    epoch_outputs[step]["metrics"]["auc"]
                )
                # Register time
                epoch_metrics[f"{step}_time"].append(epoch_outputs[step]["time"])
                # Accumulate outcomes to global metrics calculation
                for outcome in outcomes:
                    training_outcomes[step][outcome] += (
                        epoch_outputs[step]["outcomes"][outcome]
                        if epoch_outputs[step]["outcomes"][outcome] != ""
                        else 0
                    )

            # Report metrics to console
            metrics_msg = f"Epoch {epoch + 1}/{self.epochs}"
            # Train metrics
            metrics_msg += f", train loss: {train_outputs['loss']:.8f}, train F1-Score: {train_outputs['metrics']['f1score']:.8f}, train mIoU: {train_outputs['metrics']['m_iou']:.8f}, train AUC: {train_outputs['metrics']['auc']:.8f}, train time: {train_outputs['time']-epoch_start_time:.2f}"
            # Valid metrics
            metrics_msg += f", valid loss: {valid_outputs['loss']:.8f}, valid F1-Score: {valid_outputs['metrics']['f1score']:.8f}, valid mIoU: {valid_outputs['metrics']['m_iou']:.8f}, valid AUC: {valid_outputs['metrics']['auc']:.8f}, valid time: {valid_outputs['time']-train_outputs['time']:.2f}"

            # Accumulate outcomes to calculate global metrics at end of training
            # Save weights every 20 epochs and perform test step to validate
            if (epoch + 1) % save_every_epochs == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.weights_path, f"weights_{epoch+1}_epochs.pth"),
                )
                # Test metrics
                metrics_msg += f", test loss: {test_outputs['loss']:.8f}, test F1-Score: {test_outputs['metrics']['f1score']:.8f}, test mIoU: {test_outputs['metrics']['m_iou']:.8f}, test AUC: {test_outputs['metrics']['auc']:.8f}, test time: {test_outputs['time']-valid_outputs['time']:.2f}"
            # Print log message
            print(metrics_msg)
            # Save metrics to CSV on each epoch
            # Batch metrics
            batch_train_metrics_df = pd.DataFrame.from_dict(batch_train_metrics)
            batch_train_metrics_df.to_csv(
                os.path.join(self.metrics_path, "batch_train_metrics.csv")
            )
            batch_valid_metrics_df = pd.DataFrame.from_dict(batch_valid_metrics)
            batch_valid_metrics_df.to_csv(
                os.path.join(self.metrics_path, "batch_valid_metrics.csv")
            )
            batch_test_metrics_df = pd.DataFrame.from_dict(batch_test_metrics)
            batch_test_metrics_df.to_csv(
                os.path.join(self.metrics_path, "batch_test_metrics.csv")
            )
            # Epoch metrics
            epoch_metrics_df = pd.DataFrame.from_dict(epoch_metrics)
            epoch_metrics_df.to_csv(
                os.path.join(self.metrics_path, "epoch_metrics.csv")
            )
        # Calculate global losses
        training_losses["train"] /= self.epochs
        training_losses["valid"] /= self.epochs
        if (self.epochs + 1) // save_every_epochs > 0:
            training_losses["test"] /= (self.epochs + 1) // save_every_epochs
        # Calculate global metrics for process using total outcomes
        for step in ["train", "valid", "test"]:
            # Register training time
            training_metrics[f"{step}_time"].append(time.time())
            # Register losses
            training_metrics[f"{step}_loss"].append(training_losses[step])
            # Register training outcomes
            for outcome in outcomes:
                training_metrics[f"{step}_{outcome}"].append(
                    training_outcomes[step][outcome]
                )
            # Calculate metrics
            step_metrics = calculate_metrics(training_outcomes[step])
            for metric in metrics:
                training_metrics[f"{step}_{metric}"].append(step_metrics[metric])
        # Save global metrics to CSV
        training_metrics_df = pd.DataFrame.from_dict(training_metrics)
        training_metrics_df.to_csv(
            os.path.join(self.metrics_path, "training_metrics.csv")
        )

        endTime = time.time()
        logging.info("--== Training end ==--")
        logging.info(
            "Total time training and validation: {:.2f}s".format(endTime - startTime)
        )
        print("--== Training end ==--")
        print("Total time training and validation: {:.2f}s".format(endTime - startTime))
