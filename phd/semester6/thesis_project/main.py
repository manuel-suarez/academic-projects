import os
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam

home_dir = os.path.expanduser('~')

work_dir = os.path.join(home_dir, 'projects', 'cimat', '6to_semestre', 'thesis-project')
results_dir = os.path.join(work_dir, 'results')
figures_dir = os.path.join(results_dir, 'figures')
weights_dir = os.path.join(results_dir, 'weights')
metrics_dir = os.path.join(results_dir, 'metrics')
plots_dir = os.path.join(results_dir, 'plots')

for path in [figures_dir, weights_dir, metrics_dir, plots_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

batch_size = 8
num_epochs = 50

# Use library implementation
from focal_loss import BinaryFocalLoss

from models.metrics import dice_coef, jacard_coef
from models.wunet import UNet, Attention_UNet, Attention_ResUNet
from datasets.microscopy import CreateDatasets
from utils.plots import plot_history

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
def train_model(model, optimizer, loss, metrics, epochs, model_name):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # print(model.summary())

    train_dataloader, val_dataloader = CreateDatasets()

    start1 = datetime.now()
    # Using datasets and dataloaders
    model_history = model.fit(train_dataloader,
                              verbose=1,
                              batch_size=batch_size,
                              validation_data=val_dataloader,
                              shuffle=False,
                              epochs=epochs)
    stop1 = datetime.now()
    # Execution time of the model
    execution_time_Unet = stop1 - start1
    print(f"{model_name} execution time is: ", execution_time_Unet)

    # Save model
    fname = '-'.join(model_name.split(' '))
    model.save(os.path.join(weights_dir, f"mitochondria_{fname}_50epochs_B_focal.hdf5"))
    # Save history
    model_history_df = DataFrame(model_history.history)
    with open(os.path.join(metrics_dir, f"{fname}_history_df.csv"), mode='w') as f:
        model_history_df.to_csv(f)
    # Plot training loss and metrics
    plot_history(model_history, plots_dir, fname)

    # Save segmentation results
    # Load one model at a time for testing.
    model_path = os.path.join(weights_dir, f"mitochondria_{fname}_50epochs_B_focal.hdf5")

if __name__ == '__main__':
    unet_model = Attention_ResUNet(input_shape)
    models = [unet_model]
    names = ['Wavelet_Attention_ResUNet']
    for model, name in zip(models, names):
        train_model(model,
                    optimizer=Adam(learning_rate=1e-2),
                    loss=BinaryFocalLoss(gamma=2),
                    metrics=['accuracy', dice_coef, jacard_coef],
                    epochs=50,
                    model_name=name)
