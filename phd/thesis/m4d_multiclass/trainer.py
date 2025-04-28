import os
import time
import torch
import logging
import pandas as pd

from itertools import product
from calculate_outcomes import outcomes, calculate_outcomes
from calculate_metrics import metrics, calculate_metrics, calculate_roc_metrics

# Define ROC and AUC thresholds
thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def initialize_metrics(
    initial_metrics, num_classes, steps, roc_metrics=False, thresholds=thresholds
):
    for step in steps:
        # Loss
        step_loss = {f"{step}_loss": []}
        initial_metrics.update(step_loss)
        # Outcomes per classes
        for num_class in range(num_classes):
            step_class_outcomes = {
                f"{step}_class{num_class}_{outcome}": [] for outcome in outcomes
            }
            initial_metrics.update(step_class_outcomes)
        # Metrics per classes
        for num_class in range(num_classes):
            step_metrics = {
                f"{step}_class{num_class}_{metric}": [] for metric in metrics
            }
            initial_metrics.update(step_metrics)
        # Mean metrics
        mean_metrics = {f"{step}_mean_{metric}": [] for metric in metrics}
        initial_metrics.update(mean_metrics)
        # Training time
        step_time = {f"{step}_time": []}
        initial_metrics.update(step_time)
        # ROC and AUC per classes (only for valid and test steps)
        if roc_metrics:
            for num_class in range(num_classes):
                threshold_tpr_metrics = {
                    f"{step}_class{num_class}_{threshold}_tpr": []
                    for threshold in thresholds
                }
                initial_metrics.update(threshold_tpr_metrics)
                threshold_fpr_metrics = {
                    f"{step}_class{num_class}_{threshold}_fpr": []
                    for threshold in thresholds
                }
                initial_metrics.update(threshold_fpr_metrics)
                auc_metric = {f"{step}_class{num_class}_auc": []}
                initial_metrics.update(auc_metric)
            auc_mean_metric = {f"{step}_mean_auc": []}
            initial_metrics.update(auc_mean_metric)
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
        num_classes,
        epoch,
        batch_metrics,
        outcomes=outcomes,
        metrics=metrics,
        thresholds=thresholds,
    ):
        train_loss = 0.0
        epoch_class_outcomes = {
            f"class{num_class}_{outcome}": 0
            for num_class, outcome in product(range(num_classes), outcomes)
        }
        roc_class_outcomes = {
            threshold: {
                f"class{num_class}_{outcome}": 0
                for num_class, outcome in product(range(num_classes), outcomes)
            }
            for threshold in thresholds
        }
        self.model.train()
        for index, (image, label) in enumerate(dataloader):
            image, label = image.to(self.device), label.to(self.device)
            # print("Image shape: ", image.shape, image.dtype)
            # print("Label shape: ", label.shape, label.dtype)
            self.optimizer.zero_grad()
            # Get predictions
            output = self.model(image)
            # print("Output shape: ", output.shape, output.dtype)
            # Calculate loss and metrics
            loss = self.loss_fn(output, label)
            # Get prediction outcomes per classes
            train_outcomes = calculate_outcomes(
                output, label, num_classes=num_classes, use_argmax=True
            )
            # Get ROC/AUC outcomes calculation
            for threshold in thresholds:
                # Get threshold outcomes
                threshold_outcomes = calculate_outcomes(
                    output,
                    label,
                    num_classes=num_classes,
                    use_argmax=False,
                    p_threshold=threshold,
                )
                # Accumulate
                for outcome, num_class in product(outcomes, range(num_classes)):
                    roc_class_outcomes[threshold][
                        f"class{num_class}_{outcome}"
                    ] += threshold_outcomes[f"class{num_class}_{outcome}"]

            # Get prediction metrics
            train_metrics = calculate_metrics(train_outcomes, num_classes)
            # Update training
            loss.backward()
            self.optimizer.step()
            # Register batch loss, outcomes and metrics
            batch_metrics["epoch"].append(epoch)
            batch_metrics["batch"].append(index + 1)
            batch_metrics["train_loss"].append(loss.item())
            batch_metrics["start_time"].append("")
            batch_metrics["train_time"].append("")
            for outcome, num_class in product(outcomes, range(num_classes)):
                batch_metrics[f"train_class{num_class}_{outcome}"].append(
                    train_outcomes[f"class{num_class}_{outcome}"]
                )
                # Accumulate epoch outcomes for epoch metrics
                epoch_class_outcomes[f"class{num_class}_{outcome}"] += train_outcomes[
                    f"class{num_class}_{outcome}"
                ]
            mean_metrics = {f"train_mean_{metric}": 0 for metric in metrics}
            for metric, num_class in product(metrics, range(num_classes)):
                batch_metrics[f"train_class{num_class}_{metric}"].append(
                    train_metrics[f"class{num_class}_{metric}"]
                )
                mean_metrics[f"train_mean_{metric}"] += train_metrics[
                    f"class{num_class}_{metric}"
                ]
            # Calculate mean metrics
            for metric in metrics:
                mean_metrics[f"train_mean_{metric}"] /= num_classes
                batch_metrics[f"train_mean_{metric}"].append(
                    mean_metrics[f"train_mean_{metric}"]
                )

            train_loss += loss.item()

        train_loss = train_loss / len(dataloader)
        # Calculate metrics with epoch outcomes (the sum of batch outcomes instead of the mean of batch metrics)
        train_epoch_metrics = calculate_metrics(epoch_class_outcomes, num_classes)
        # Calculate mean metrics per epoch outcomes
        for metric in metrics:
            train_epoch_metrics[f"mean_{metric}"] = 0
            for num_class in range(num_classes):
                train_epoch_metrics[f"mean_{metric}"] += train_epoch_metrics[
                    f"class{num_class}_{metric}"
                ]
            train_epoch_metrics[f"mean_{metric}"] /= num_classes
        # Accumulate values
        tpr_class_values = {num_class: [] for num_class in range(num_classes)}
        fpr_class_values = {num_class: [] for num_class in range(num_classes)}
        # Calculate ROC per class
        for num_class in range(num_classes):
            tpr_class_values[num_class].append(0.0)
            fpr_class_values[num_class].append(0.0)
            for threshold in thresholds:
                threshold_metrics = calculate_roc_metrics(
                    roc_class_outcomes[threshold], num_classes
                )
                threshold_tpr = threshold_metrics[f"class{num_class}_tpr"]
                threshold_fpr = threshold_metrics[f"class{num_class}_fpr"]
                # Store in metrics
                train_epoch_metrics[f"class{num_class}_{threshold}_tpr"] = threshold_tpr
                train_epoch_metrics[f"class{num_class}_{threshold}_fpr"] = threshold_fpr
                # Accumulcate TPR/FPR for AUC calculation
                tpr_class_values[num_class].append(threshold_tpr)
                fpr_class_values[num_class].append(threshold_fpr)
            # Calculate AUC
            auc_value = 0.0
            for i in range(1, len(fpr_class_values[num_class])):
                auc_value += (
                    (
                        fpr_class_values[num_class][i]
                        - fpr_class_values[num_class][i - 1]
                    )
                    * (
                        tpr_class_values[num_class][i]
                        + tpr_class_values[num_class][i - 1]
                    )
                    / 2
                )
            train_epoch_metrics[f"class{num_class}_auc"] = auc_value
        # Calculate mean AUC
        train_epoch_metrics["mean_auc"] = 0
        for num_class in range(num_classes):
            train_epoch_metrics["mean_auc"] += train_epoch_metrics[
                f"class{num_class}_auc"
            ]
        train_epoch_metrics["mean_auc"] /= num_classes

        return {
            "loss": train_loss,
            "outcomes": epoch_class_outcomes,
            "metrics": train_epoch_metrics,
            "time": time.time(),
        }

    def valid_step(
        self,
        dataloader,
        num_classes,
        epoch,
        batch_metrics,
        outcomes=outcomes,
        metrics=metrics,
        thresholds=thresholds,
    ):
        valid_loss = 0.0
        epoch_class_outcomes = {
            f"class{num_class}_{outcome}": 0
            for num_class, outcome in product(range(num_classes), outcomes)
        }
        roc_class_outcomes = {
            threshold: {
                f"class{num_class}_{outcome}": 0
                for num_class, outcome in product(range(num_classes), outcomes)
            }
            for threshold in thresholds
        }
        self.model.eval()
        for index, (image, label) in enumerate(dataloader):
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                loss = self.loss_fn(output, label)
                # Get prediction outcomes per classes
                valid_outcomes = calculate_outcomes(
                    output, label, num_classes=num_classes, use_argmax=True
                )
                # Get ROC/AUC outcomes calculation
                for threshold in thresholds:
                    # Get threshold outcomes
                    threshold_outcomes = calculate_outcomes(
                        output,
                        label,
                        num_classes=num_classes,
                        use_argmax=False,
                        p_threshold=threshold,
                    )
                    # Accumulate
                    for outcome, num_class in product(outcomes, range(num_classes)):
                        roc_class_outcomes[threshold][
                            f"class{num_class}_{outcome}"
                        ] += threshold_outcomes[f"class{num_class}_{outcome}"]

                # Get prediction metrics
                valid_metrics = calculate_metrics(valid_outcomes, num_classes)
                # Register batch loss, outcomes and metrics
                batch_metrics["epoch"].append(epoch)
                batch_metrics["batch"].append(index + 1)
                batch_metrics["valid_loss"].append(loss.item())
                batch_metrics["start_time"].append("")
                batch_metrics["valid_time"].append("")
                for outcome, num_class in product(outcomes, range(num_classes)):
                    batch_metrics[f"valid_class{num_class}_{outcome}"].append(
                        valid_outcomes[f"class{num_class}_{outcome}"]
                    )
                    # Accumulate epoch outcomes for epoch metrics
                    epoch_class_outcomes[
                        f"class{num_class}_{outcome}"
                    ] += valid_outcomes[f"class{num_class}_{outcome}"]
                mean_metrics = {f"valid_mean_{metric}": 0 for metric in metrics}
                for metric, num_class in product(metrics, range(num_classes)):
                    batch_metrics[f"valid_class{num_class}_{metric}"].append(
                        valid_metrics[f"class{num_class}_{metric}"]
                    )
                    mean_metrics[f"valid_mean_{metric}"] += valid_metrics[
                        f"class{num_class}_{metric}"
                    ]
                # Calculate mean metrics
                for metric in metrics:
                    mean_metrics[f"valid_mean_{metric}"] /= num_classes
                    batch_metrics[f"valid_mean_{metric}"].append(
                        mean_metrics[f"valid_mean_{metric}"]
                    )

                valid_loss += loss.item()

        valid_loss = valid_loss / len(dataloader)
        # Calculate metrics with epoch outcomes (the sum of batch outcomes instead of the mean of batch metrics)
        valid_epoch_metrics = calculate_metrics(epoch_class_outcomes, num_classes)
        # Calculate mean metrics per epoch outcomes
        for metric in metrics:
            valid_epoch_metrics[f"mean_{metric}"] = 0
            for num_class in range(num_classes):
                valid_epoch_metrics[f"mean_{metric}"] += valid_epoch_metrics[
                    f"class{num_class}_{metric}"
                ]
            valid_epoch_metrics[f"mean_{metric}"] /= num_classes
        # Accumulate values
        tpr_class_values = {num_class: [] for num_class in range(num_classes)}
        fpr_class_values = {num_class: [] for num_class in range(num_classes)}
        # Calculate ROC per class
        for num_class in range(num_classes):
            tpr_class_values[num_class].append(0.0)
            fpr_class_values[num_class].append(0.0)
            for threshold in thresholds:
                threshold_metrics = calculate_roc_metrics(
                    roc_class_outcomes[threshold], num_classes
                )
                threshold_tpr = threshold_metrics[f"class{num_class}_tpr"]
                threshold_fpr = threshold_metrics[f"class{num_class}_fpr"]
                # Store in metrics
                valid_epoch_metrics[f"class{num_class}_{threshold}_tpr"] = threshold_tpr
                valid_epoch_metrics[f"class{num_class}_{threshold}_fpr"] = threshold_fpr
                # Accumulcate TPR/FPR for AUC calculation
                tpr_class_values[num_class].append(threshold_tpr)
                fpr_class_values[num_class].append(threshold_fpr)
            # Calculate AUC
            auc_value = 0.0
            for i in range(1, len(fpr_class_values[num_class])):
                auc_value += (
                    (
                        fpr_class_values[num_class][i]
                        - fpr_class_values[num_class][i - 1]
                    )
                    * (
                        tpr_class_values[num_class][i]
                        + tpr_class_values[num_class][i - 1]
                    )
                    / 2
                )
            valid_epoch_metrics[f"class{num_class}_auc"] = auc_value
        # Calculate mean AUC
        valid_epoch_metrics["mean_auc"] = 0
        for num_class in range(num_classes):
            valid_epoch_metrics["mean_auc"] += valid_epoch_metrics[
                f"class{num_class}_auc"
            ]
        valid_epoch_metrics["mean_auc"] /= num_classes

        return {
            "loss": valid_loss,
            "outcomes": epoch_class_outcomes,
            "metrics": valid_epoch_metrics,
            "time": time.time(),
        }

    def test_step(
        self,
        dataloader,
        num_classes,
        epoch,
        batch_metrics,
        outcomes=outcomes,
        metrics=metrics,
        thresholds=thresholds,
    ):
        test_loss = 0.0
        epoch_class_outcomes = {
            f"class{num_class}_{outcome}": 0
            for num_class, outcome in product(range(num_classes), outcomes)
        }
        roc_class_outcomes = {
            threshold: {
                f"class{num_class}_{outcome}": 0
                for num_class, outcome in product(range(num_classes), outcomes)
            }
            for threshold in thresholds
        }
        self.model.eval()
        for index, (image, label) in enumerate(dataloader):
            image, label = image.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(image)
                loss = self.loss_fn(output, label)
                # Get prediction outcomes per classes
                test_outcomes = calculate_outcomes(
                    output, label, num_classes=num_classes, use_argmax=True
                )
                # Get ROC/AUC outcomes calculation
                for threshold in thresholds:
                    # Get threshold outcomes
                    threshold_outcomes = calculate_outcomes(
                        output,
                        label,
                        num_classes=num_classes,
                        use_argmax=False,
                        p_threshold=threshold,
                    )
                    # Accumulate
                    for outcome, num_class in product(outcomes, range(num_classes)):
                        roc_class_outcomes[threshold][
                            f"class{num_class}_{outcome}"
                        ] += threshold_outcomes[f"class{num_class}_{outcome}"]

                # Get prediction metrics
                test_metrics = calculate_metrics(test_outcomes, num_classes)
                # Register batch loss, outcomes and metrics
                batch_metrics["epoch"].append(epoch)
                batch_metrics["batch"].append(index + 1)
                batch_metrics["test_loss"].append(loss.item())
                batch_metrics["start_time"].append("")
                batch_metrics["test_time"].append("")
                for outcome, num_class in product(outcomes, range(num_classes)):
                    batch_metrics[f"test_class{num_class}_{outcome}"].append(
                        test_outcomes[f"class{num_class}_{outcome}"]
                    )
                    # Accumulate epoch outcomes for epoch metrics
                    epoch_class_outcomes[
                        f"class{num_class}_{outcome}"
                    ] += test_outcomes[f"class{num_class}_{outcome}"]
                mean_metrics = {f"test_mean_{metric}": 0 for metric in metrics}
                for metric, num_class in product(metrics, range(num_classes)):
                    batch_metrics[f"test_class{num_class}_{metric}"].append(
                        test_metrics[f"class{num_class}_{metric}"]
                    )
                    mean_metrics[f"test_mean_{metric}"] += test_metrics[
                        f"class{num_class}_{metric}"
                    ]
                # Calculate mean metrics
                for metric in metrics:
                    mean_metrics[f"test_mean_{metric}"] /= num_classes
                    batch_metrics[f"test_mean_{metric}"].append(
                        mean_metrics[f"test_mean_{metric}"]
                    )

                test_loss += loss.item()

        test_loss = test_loss / len(dataloader)
        # Calculate metrics with epoch outcomes (the sum of batch outcomes instead of the mean of batch metrics)
        test_epoch_metrics = calculate_metrics(epoch_class_outcomes, num_classes)
        # Calculate mean metrics per epoch outcomes
        for metric in metrics:
            test_epoch_metrics[f"mean_{metric}"] = 0
            for num_class in range(num_classes):
                test_epoch_metrics[f"mean_{metric}"] += test_epoch_metrics[
                    f"class{num_class}_{metric}"
                ]
            test_epoch_metrics[f"mean_{metric}"] /= num_classes
        # Accumulate values
        tpr_class_values = {num_class: [] for num_class in range(num_classes)}
        fpr_class_values = {num_class: [] for num_class in range(num_classes)}
        # Calculate ROC per class
        for num_class in range(num_classes):
            tpr_class_values[num_class].append(0.0)
            fpr_class_values[num_class].append(0.0)
            for threshold in thresholds:
                threshold_metrics = calculate_roc_metrics(
                    roc_class_outcomes[threshold], num_classes
                )
                threshold_tpr = threshold_metrics[f"class{num_class}_tpr"]
                threshold_fpr = threshold_metrics[f"class{num_class}_fpr"]
                # Store in metrics
                test_epoch_metrics[f"class{num_class}_{threshold}_tpr"] = threshold_tpr
                test_epoch_metrics[f"class{num_class}_{threshold}_fpr"] = threshold_fpr
                # Accumulcate TPR/FPR for AUC calculation
                tpr_class_values[num_class].append(threshold_tpr)
                fpr_class_values[num_class].append(threshold_fpr)
            # Calculate AUC
            auc_value = 0.0
            for i in range(1, len(fpr_class_values[num_class])):
                auc_value += (
                    (
                        fpr_class_values[num_class][i]
                        - fpr_class_values[num_class][i - 1]
                    )
                    * (
                        tpr_class_values[num_class][i]
                        + tpr_class_values[num_class][i - 1]
                    )
                    / 2
                )
            test_epoch_metrics[f"class{num_class}_auc"] = auc_value
        # Calculate mean AUC
        test_epoch_metrics["mean_auc"] = 0
        for num_class in range(num_classes):
            test_epoch_metrics["mean_auc"] += test_epoch_metrics[
                f"class{num_class}_auc"
            ]
        test_epoch_metrics["mean_auc"] /= num_classes

        return {
            "loss": test_loss,
            "outcomes": epoch_class_outcomes,
            "metrics": test_epoch_metrics,
            "time": time.time(),
        }

    # Fake step to get empty values for test step
    def zeros_step(
        self, num_classes, outcomes=outcomes, metrics=metrics, thresholds=thresholds
    ):
        zeros_metrics = {
            f"class{num_class}_{metric}": ""
            for num_class, metric in product(range(num_classes), metrics)
        }
        mean_metrics = {f"mean_{metric}": "" for metric in metrics}
        zeros_metrics.update(mean_metrics)
        for num_class, threshold in product(range(num_classes), thresholds):
            zeros_metrics[f"class{num_class}_{threshold}_tpr"] = ""
            zeros_metrics[f"class{num_class}_{threshold}_fpr"] = ""
        for num_class in range(num_classes):
            zeros_metrics[f"class{num_class}_auc"] = ""
        zeros_metrics["mean_auc"] = ""
        return {
            "loss": "",
            "outcomes": {
                f"class{num_class}_{outcome}": ""
                for num_class, outcome in product(range(num_classes), outcomes)
            },
            "metrics": zeros_metrics,
            "time": "",
        }

    def fit(
        self,
        dataloaders,
        num_classes,
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
            {"epoch": [], "batch": [], "start_time": []},
            num_classes,
            ["train"],
            roc_metrics=False,
        )
        batch_valid_metrics = initialize_metrics(
            {"epoch": [], "batch": [], "start_time": []},
            num_classes,
            ["valid"],
            roc_metrics=False,
        )
        batch_test_metrics = initialize_metrics(
            {"epoch": [], "batch": [], "start_time": []},
            num_classes,
            ["test"],
            roc_metrics=False,
        )
        # Initialize dictionary for epoch metrics
        epoch_metrics = initialize_metrics(
            {"epoch": [], "start_time": []},
            num_classes,
            ["train", "valid", "test"],
            roc_metrics=True,
        )
        # Initialize dictionary for training process metrics
        training_metrics = initialize_metrics(
            {"start_time": []},
            num_classes,
            ["train", "valid", "test"],
            roc_metrics=False,
        )
        training_losses = {"train": 0.0, "valid": 0.0, "test": 0.0}
        # Initialize global outcomes
        training_outcomes = {
            f"{step}": {
                f"class{num_class}_{outcome}": 0
                for outcome, num_class in product(outcomes, range(num_classes))
            }
            for step in ["train", "valid", "test"]
        }

        logging.info("--== Training start ==--")
        print("--== Training start ==--")
        startTime = time.time()

        training_metrics["start_time"].append(time.time())
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_outputs = self.train_step(
                train_dataloader, num_classes, epoch, batch_train_metrics
            )
            valid_outputs = self.valid_step(
                valid_dataloader, num_classes, epoch, batch_valid_metrics
            )
            # Run test step each 20 epochs
            if (epoch + 1) % 20 == 0:
                # Perform test step
                test_outputs = self.test_step(
                    test_dataloader, num_classes, epoch, batch_test_metrics
                )
            else:
                test_outputs = self.zeros_step(num_classes)

            epoch_outputs = {
                "train": train_outputs,
                "valid": valid_outputs,
                "test": test_outputs,
            }
            # Register outcomes and metrics
            epoch_metrics["epoch"].append(epoch + 1)
            epoch_metrics["start_time"].append(epoch_start_time)
            # Register epoch loss, outcomes, and metrics
            for step in ["train", "valid", "test"]:
                # Register loss
                epoch_metrics[f"{step}_loss"].append(epoch_outputs[step]["loss"])
                training_losses[f"{step}"] += (
                    0
                    if step == "test" and (epoch + 1) % 20 != 0
                    else epoch_outputs[step]["loss"]
                )
                # Register outcomes
                for outcome, num_class in product(outcomes, range(num_classes)):
                    epoch_metrics[f"{step}_class{num_class}_{outcome}"].append(
                        epoch_outputs[step]["outcomes"][f"class{num_class}_{outcome}"]
                    )
                # Register metrics
                for metric, num_class in product(metrics, range(num_classes)):
                    epoch_metrics[f"{step}_class{num_class}_{metric}"].append(
                        epoch_outputs[step]["metrics"][f"class{num_class}_{metric}"]
                    )
                # Register mean metrics
                for metric in metrics:
                    epoch_metrics[f"{step}_mean_{metric}"].append(
                        epoch_outputs[step]["metrics"][f"mean_{metric}"]
                    )
                # Register ROC per threshold
                for threshold, num_class in product(thresholds, range(num_classes)):
                    epoch_metrics[f"{step}_class{num_class}_{threshold}_tpr"].append(
                        epoch_outputs[step]["metrics"][
                            f"class{num_class}_{threshold}_tpr"
                        ]
                    )
                    epoch_metrics[f"{step}_class{num_class}_{threshold}_fpr"].append(
                        epoch_outputs[step]["metrics"][
                            f"class{num_class}_{threshold}_fpr"
                        ]
                    )
                # Register AUC
                for num_class in range(num_classes):
                    epoch_metrics[f"{step}_class{num_class}_auc"].append(
                        epoch_outputs[step]["metrics"][f"class{num_class}_auc"]
                    )
                # Register mean AUC
                epoch_metrics[f"{step}_mean_auc"].append(
                    epoch_outputs[step]["metrics"]["mean_auc"]
                )
                # Register time
                epoch_metrics[f"{step}_time"].append(epoch_outputs[step]["time"])
                # Accumulate outcomes to global metrics calculation
                for outcome, num_class in product(outcomes, range(num_classes)):
                    training_outcomes[step][f"class{num_class}_{outcome}"] += (
                        epoch_outputs[step]["outcomes"][f"class{num_class}_{outcome}"]
                        if epoch_outputs[step]["outcomes"][
                            f"class{num_class}_{outcome}"
                        ]
                        != ""
                        else 0
                    )

            # Report metrics to console
            metrics_msg = f"Epoch {epoch + 1}/{self.epochs}"
            # Train metrics
            metrics_msg += f", train loss: {train_outputs['loss']:.8f}, train mean accuracy: {train_outputs['metrics']['mean_accuracy']:.8f}, train mean IoU: {train_outputs['metrics']['mean_f_iou']:.8f}, train mean AUC: {train_outputs['metrics']['mean_auc']:.8f}, train time: {train_outputs['time']-epoch_start_time:.2f}"
            # Valid metrics
            metrics_msg += f", valid loss: {valid_outputs['loss']:.8f}, valid mean accuracy: {valid_outputs['metrics']['mean_accuracy']:.8f}, valid mean iou: {valid_outputs['metrics']['mean_f_iou']:.8f}, valid mean AUC: {valid_outputs['metrics']['mean_auc']:.8f}, valid time: {valid_outputs['time']-train_outputs['time']:.2f}"

            # Save weights every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.weights_path, f"weights_{epoch+1}_epochs.pth"),
                )
                # Test metrics
                metrics_msg += f", test loss: {test_outputs['loss']:.8f}, test mean accuracy: {test_outputs['metrics']['mean_accuracy']:.8f}, test mean IoU: {test_outputs['metrics']['mean_f_iou']:.8f}, test mean AUC: {test_outputs['metrics']['mean_auc']:.8f}, test time: {test_outputs['time']-valid_outputs['time']:.2f}"

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
        if (self.epochs + 1) // 20 > 0:
            training_losses["test"] /= (self.epochs + 1) // 20
        # Calculate global metrics for process using total outcomes
        for step in ["train", "valid", "test"]:
            # Register training time
            training_metrics[f"{step}_time"].append(time.time())
            # Register losses
            training_metrics[f"{step}_loss"].append(training_losses[step])
            # Register training outcomes
            for outcome, num_class in product(outcomes, range(num_classes)):
                training_metrics[f"{step}_class{num_class}_{outcome}"].append(
                    training_outcomes[step][f"class{num_class}_{outcome}"]
                )
            # Calculate metrics
            step_metrics = calculate_metrics(training_outcomes[step], num_classes)
            mean_metrics = {f"{step}_mean_{metric}": 0 for metric in metrics}
            for metric, num_class in product(metrics, range(num_classes)):
                training_metrics[f"{step}_class{num_class}_{metric}"].append(
                    step_metrics[f"class{num_class}_{metric}"]
                )
                mean_metrics[f"{step}_mean_{metric}"] += step_metrics[
                    f"class{num_class}_{metric}"
                ]
            # Calculate mean metrics
            for metric in metrics:
                mean_metrics[f"{step}_mean_{metric}"] /= num_classes
                training_metrics[f"{step}_mean_{metric}"].append(
                    mean_metrics[f"{step}_mean_{metric}"]
                )
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
