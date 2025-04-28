import os
import torch
import logging
import argparse
import pandas as pd

from itertools import product
from models import validate_model, get_model_identifier
from models.encoders import validate_encoder, validate_layer, get_encoder_identifier

# In this scripts we are generating the latex code for different comparative tables
# of the metrics obtained with the models


# Because we have different variables of metrics we can combine them in different ways
# Variables:
# - Model: semantic segmentation models (defined by the decoder part of the model)
#   - UNet, UNet2p, UNet3p, LinkNet, PSPNet, FPNet, DeepLabV3p, MA-Net
# - Encoder: architecture used in the encoder part of the model
#   - ResNet, SENet, CBAMNet, MRNet
# - Layers: size of the encoder in terms of the layers used
#   - 18, 34, 50, 101, 152
# - Epochs: num of epochs of the training process
#   - 1-100
# - Metrics: metrics calculated in the training and validation process
#   - Accuracy, Precision, Recall, Sensitivity, F1-Score, IoU, Dice
def list_of_strings(arg):
    return arg.split(",")


# Validations
def list_of_models(arg):
    models = list_of_strings(arg)
    if not all([validate_model(model) for model in models]):
        not_valid_models = [model for model in models if not validate_model(model)]

        raise argparse.ArgumentTypeError(
            f"Model's \"{', '.join(not_valid_models)}\" are not valid models specification"
        )
    count_models = {model: models.count(model) for model in models}
    print("Count models: ", count_models)
    print("Validate: ", [count > 1 for count in count_models.values()])
    if any([count > 1 for count in count_models.values()]):
        not_valid_models = [model for model in models if count_models[model] > 1]
        print("Not valid models: ", not_valid_models)

        raise argparse.ArgumentTypeError(
            f"Model's \"{', '.join(set(not_valid_models))}\" were specified more than one time"
        )
    return models


def list_of_encoders(arg):
    encoders = list_of_strings(arg)
    if not all([validate_encoder(encoder) for encoder in encoders]):
        not_valid_encoders = [
            encoder for encoder in encoders if not validate_encoder(encoder)
        ]

        raise argparse.ArgumentTypeError(
            f"Encoder's \"{', '.join(not_valid_encoders)}\" are not valid encoders specification"
        )
    count_encoders = {encoder: encoders.count(encoder) for encoder in encoders}
    print("Count encoders: ", count_encoders)
    print("Validate: ", [count > 1 for count in count_encoders.values()])
    if any([count > 1 for count in count_encoders.values()]):
        not_valid_encoders = [
            encoder for encoder in encoders if count_encoders[encoder] > 1
        ]
        print("Not valid encoders: ", not_valid_encoders)

        raise argparse.ArgumentTypeError(
            f"encoder's \"{', '.join(set(not_valid_encoders))}\" were specified more than one time"
        )
    return encoders


def list_of_layers(arg):
    layers = list_of_strings(arg)
    if not all([validate_layer(layer) for layer in layers]):
        not_valid_layers = [layer for layer in layers if not validate_layer(layer)]

        raise argparse.ArgumentTypeError(
            f"Layer's \"{', '.join(not_valid_layers)}\" are not valid layers specification"
        )
    count_layers = {layer: layers.count(layer) for layer in layers}
    print("Count layers: ", count_layers)
    print("Validate: ", [count > 1 for count in count_layers.values()])
    if any([count > 1 for count in count_layers.values()]):
        not_valid_layers = [layer for layer in layers if count_layers[layer] > 1]
        print("Not valid layers: ", not_valid_layers)

        raise argparse.ArgumentTypeError(
            f"layer's \"{', '.join(set(not_valid_layers))}\" were specified more than one time"
        )
    return layers


parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    type=list_of_models,
    required=True,
    default="unet,unet2p,unet3p",
    help="comma separated list of models to evaluate",
)
parser.add_argument(
    "--encoders",
    type=list_of_encoders,
    required=True,
    default="resnet,senet,cbamnet",
    help="comma separated list of encoders to evaluate",
)
parser.add_argument(
    "--layers",
    type=list_of_layers,
    required=True,
    default="18,34,50",
    help="comma separated list of encoders num of layers",
)
parser.add_argument(
    "--epochs", type=int, required=True, default=60, help="epochs to analyze"
)
parser.add_argument(
    "--metrics",
    type=list_of_strings,
    required=True,
    default="accuracy,precision,recall,specificity,iou,dice",
    help="comma separated list of metrics to evaluate",
)
parser.add_argument(
    "--steps",
    type=list_of_strings,
    required=True,
    default="train,valid,test",
    help="comma separated list of ",
)
parser.add_argument(
    "--metrics_path",
    type=str,
    required=False,
    default="metrics",
    help="path of metrics files",
)
parser.add_argument(
    "--metrics_file",
    type=str,
    required=False,
    default="epoch_metrics.csv",
    help="file name of metrics to load",
)
parser.add_argument(
    "--operation_mode",
    type=str,
    required=True,
    default="per_model_encoder",
    help="to specify the type of table to generate",
)
args = parser.parse_args()
print(args)
epochs = args.epochs

# Model, Encoder and Layers metrics are saved per file according to the training process
# - Directory structure: encoder-layer/model/metrics.csv
# Epochs and metrics are saved in each row of the metrics.csv file

# So, there are different ways of organize the tables:

# Per model, comparing encoders (rows) and layers (columns) with a fixed num of epochs
# Per model, comparing encoder-layer (rows) with variable num of epochs (cols)

# We need a code base to dynamically select the structure needed for the table however
# in this first version we are using the next combination for the informs needed:

# Per model, compare metrics for encoders (rows) with num of layers (cols)

# Define variables (we are defining locally the variables to not depend on the modules)
models = args.models
encoders = args.encoders
layers = args.layers
metrics = args.metrics
# metrics = ["accuracy", "precision", "recall", "specificity", "iou", "dice"]
steps = args.steps
metrics_path = args.metrics_path
metrics_file = args.metrics_file

# Initial configuration
print("Metrics path: ", metrics_path)
print("Metrics file: ", metrics_file)

# How to organize results
operation_mode = args.operation_mode


# Print metrics by model (rows) and encoder+layer (columns)
def print_metric_results_by_model_encoder(metric, step):
    print("\\subsubsection{{Metric: {0}, step: {1}}}".format(metric, step))
    table_results = {
        "Models": [],
    }
    encoders_dicts = {
        get_encoder_identifier(encoder, layer): []
        for encoder, layer in product(encoders, layers)
    }
    table_results.update(encoders_dicts)

    calc_count = 0
    for model, encoder, layer in product(models, encoders, layers):
        if get_model_identifier(model) not in table_results["Models"]:
            table_results["Models"].append(get_model_identifier(model))
        encoder_layer = f"{encoder}{layer}"
        encoder_path = os.path.join(metrics_path, encoder_layer, model)
        # Open encoder file and read results
        encoder_file = os.path.join(encoder_path, metrics_file)
        if not os.path.exists(encoder_file):
            table_results[get_encoder_identifier(encoder, layer)].append("NA")
            continue
        # Open CSV
        metrics_data = pd.read_csv(os.path.join(encoder_path, metrics_file))
        # Check row num
        if not (metrics_data["epoch"] == epochs).any():
            table_results[get_encoder_identifier(encoder, layer)].append("NA")
            continue
        # Read row corresponding to epoch
        epoch_data = metrics_data[metrics_data["epoch"] == epochs]
        # print("Epoch data: ", epoch_data)
        # Read loss
        metric_value = epoch_data[f"{step}_{metric}"].iloc[0]
        # print("Metric value: ", model, encoder_layer, metric_value)
        # Append to model-encoder data
        table_results[get_encoder_identifier(encoder, layer)].append(metric_value)
        calc_count += 1

    # We need to have the same length on all lists of the dict
    test_len = len(table_results["Models"])
    if not all(
        [len(table_results[table_index]) == test_len for table_index in table_results]
    ):
        print(
            f"Error en el cálculo de la tabla, metric: {metric}, step: {step}, model: {model}, encoder: {get_encoder_identifier(encoder, layer)}"
        )
        print(f"Número de modelos: {len(table_results['Models'])-1}")
        for e_name in [e_name for e_name in table_results.keys() if e_name != "Model"]:
            print(
                f"Número de métricas para el encoder: {e_name}: {len(table_results[e_name])-1}"
            )
        exit(0)
    # print(table_results)
    df_results = pd.DataFrame(table_results)
    print(df_results.to_latex(index=False, float_format="{:.6f}".format))


# Print metrics by model+encoder+layer (rows) and metrics (columns)
def print_metric_results_by_model_metric(encoder, layer, step):
    print(
        "\\subsubsection{{Encoder: {0}, step: {1}}}".format(
            get_encoder_identifier(encoder, layer), step
        )
    )
    table_results = {
        "Models": [],
    }
    metrics_dicts = {metric: [] for metric in metrics}
    table_results.update(metrics_dicts)

    calc_count = 0
    for model, metric in product(models, metrics):
        if get_model_identifier(model) not in table_results["Models"]:
            table_results["Models"].append(get_model_identifier(model))
        encoder_layer = f"{encoder}{layer}"
        encoder_path = os.path.join(metrics_path, encoder_layer, model)
        # Open encoder file and read results
        encoder_file = os.path.join(encoder_path, metrics_file)
        if not os.path.exists(encoder_file):
            table_results[metric].append("NA")
            continue
        # Open CSV
        metrics_data = pd.read_csv(os.path.join(encoder_path, metrics_file))
        # Check row num
        if not (metrics_data["epoch"] == epochs).any():
            table_results[metric].append("NA")
            continue
        # Read row corresponding to epoch
        epoch_data = metrics_data[metrics_data["epoch"] == epochs]
        # print("Epoch data: ", epoch_data)
        # Read loss
        metric_value = epoch_data[f"{step}_{metric}"].iloc[0]
        # print("Metric value: ", model, encoder_layer, metric_value)
        # Append to model-encoder data
        table_results[metric].append(metric_value)
        calc_count += 1

    # We need to have the same length on all lists of the dict
    test_len = len(table_results["Models"])
    if not all(
        [len(table_results[table_index]) == test_len for table_index in table_results]
    ):
        print(
            f"Error en el cálculo de la tabla, metric: {metric}, step: {step}, model: {model}, encoder: {get_encoder_identifier(encoder, layer)}"
        )
        print(f"Número de modelos: {len(table_results['Models'])-1}")
        for e_name in [e_name for e_name in table_results.keys() if e_name != "Model"]:
            print(
                f"Número de métricas para el encoder: {e_name}: {len(table_results[e_name])-1}"
            )
        exit(0)
    # print(table_results)
    df_results = pd.DataFrame(table_results)
    print(df_results.to_latex(index=False, float_format="{:.6f}".format))


# Print metrics by epochs (rows) and metrics (columns)
def print_metric_results_by_epoch_metric(model, encoder, layer, step):
    print(
        "\\subsubsection{{Model: {0}, Encoder: {1}, step: {2}}}".format(
            get_model_identifier(model), get_encoder_identifier(encoder, layer), step
        )
    )
    table_results = {
        "Epochs": [],
    }
    metrics_dicts = {metric: [] for metric in metrics}
    table_results.update(metrics_dicts)

    calc_count = 0
    for epoch, metric in product(range(1, args.epochs + 1), metrics):
        if epoch not in table_results["Epochs"]:
            table_results["Epochs"].append(epoch)
        encoder_layer = f"{encoder}{layer}"
        encoder_path = os.path.join(metrics_path, encoder_layer, model)
        # Open encoder file and read results
        encoder_file = os.path.join(encoder_path, metrics_file)
        if not os.path.exists(encoder_file):
            table_results[metric].append("NA")
            continue
        # Open CSV
        metrics_data = pd.read_csv(os.path.join(encoder_path, metrics_file))
        # Check row num
        if not (metrics_data["epoch"] == epoch).any():
            table_results[metric].append("NA")
            continue
        # Read row corresponding to epoch
        epoch_data = metrics_data[metrics_data["epoch"] == epoch]
        # print("Epoch data: ", epoch_data)
        # Read loss
        metric_value = epoch_data[f"{step}_{metric}"].iloc[0]
        # print("Metric value: ", model, encoder_layer, metric_value)
        # Append to model-encoder data
        table_results[metric].append(metric_value)
        calc_count += 1

    # We need to have the same length on all lists of the dict
    test_len = len(table_results["Epochs"])
    if not all(
        [len(table_results[table_index]) == test_len for table_index in table_results]
    ):
        print(
            f"Error en el cálculo de la tabla, metric: {metric}, step: {step}, model: {model}, encoder: {get_encoder_identifier(encoder, layer)}"
        )
        print(f"Número de modelos: {len(table_results['Epochs'])-1}")
        for e_name in [e_name for e_name in table_results.keys() if e_name != "Epochs"]:
            print(
                f"Número de métricas para el encoder: {e_name}: {len(table_results[e_name])-1}"
            )
        exit(0)
    # print(table_results)
    df_results = pd.DataFrame(table_results)
    print(df_results.to_latex(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    if operation_mode == "per_model_encoder":
        for metric, step in product(metrics, steps):
            print_metric_results_by_model_encoder(metric, step)
    elif operation_mode == "per_model_metric":
        for encoder, layer, step in product(encoders, layers, steps):
            print_metric_results_by_model_metric(encoder, layer, step)
    elif operation_mode == "per_epoch_metric":
        for model, encoder, layer, step in product(models, encoders, layers, steps):
            print_metric_results_by_epoch_metric(model, encoder, layer, step)
    else:
        print(f"Operation mode: {operation_mode} is not a valid operation mode")
