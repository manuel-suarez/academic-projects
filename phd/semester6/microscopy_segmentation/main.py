"""
Mitochondria semantic segmentation using U-net, Attention Unet and Att Res Unet
"""

import os
from pandas import DataFrame
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam

home_dir = os.path.expanduser('~')
base_dir = os.path.join(home_dir, 'data')
work_dir = os.path.join(base_dir, 'microscopy-dataset')
data_dir = os.path.join(work_dir, 'segmentation')
results_dir = os.path.join(work_dir, 'results')
train_dir = os.path.join(work_dir, 'training2', 'train')
val_dir = os.path.join(work_dir, 'training2', 'val')
figures_dir = os.path.join(results_dir, 'figures')
weights_dir = os.path.join(results_dir, 'weights')
metrics_dir = os.path.join(results_dir, 'metrics')
plots_dir = os.path.join(results_dir, 'plots')
for path in [figures_dir, weights_dir, metrics_dir, plots_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

SIZE = 256

# Use image generators to load images from disk
seed = 24
batch_size = 8

# Parameters for model
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
num_epochs = 50

# Use library implementation
from focal_loss import BinaryFocalLoss

#from models.functional.unet import UNet
from models.implementations.models_v1 import dice_coef, jacard_coef
from models.functional.unet import UNet
from models.functional.unetpp import UNetPP
from dataset.generators import create_generators
from dataset.memory import create_datasets
from dataset.sequences import Dataset, Dataloader

def visualize(figure_name, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(figure_name)
    plt.close()

def train_model(model, optimizer, loss, metrics, epochs, model_name):
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Using sequences
    train_dataset = Dataset(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'masks'))
    val_dataset = Dataset(os.path.join(val_dir, 'images'), os.path.join(val_dir, 'masks'))
    # Visualize image
    image, mask = train_dataset[5]
    visualize('train_images.png', image=image, mask=mask)
    image, mask = val_dataset[5]
    visualize('val_images.png', image=image, mask=mask)
    train_dataloader = Dataloader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=8, shuffle=False)
    #X_train, X_test, y_train, y_test = create_datasets(image_dir, mask_dir)
    #num_train_imgs, train_generator, val_generator = create_generators(data_dir, batch_size, seed)
    start1 = datetime.now()
    # Using datasets and dataloaders
    model_history = model.fit(train_dataloader,
                              verbose=1,
                              batch_size=batch_size,
                              validation_data=val_dataloader,
                              shuffle=False,
                              epochs=epochs)
    # Using memory lists
    #model_history = model.fit(X_train, y_train,
    #                          verbose=1,
    #                          batch_size=batch_size,
    #                          validation_data=(X_test, y_test),
    #                          shuffle=False,
    #                          epochs=epochs)
    # Using generators
    #model_history = model.fit(train_generator,
    #                          verbose=1,
    #                          batch_size=batch_size,
    #                          steps_per_epoch=num_train_imgs//batch_size,
    #                          validation_data=val_generator,
    #                          validation_steps=num_train_imgs//batch_size,
    #                          shuffle=False,
    #                          epochs=epochs)
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
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{fname}_loss.png"))
    plt.close()

    # acc = history.history['jacard_coef']
    acc = model_history.history['accuracy']
    # val_acc = history.history['val_jacard_coef']
    val_acc = model_history.history['val_accuracy']

    plt.figure()
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Jacard')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{fname}_accuracy.png"))
    plt.close()

    # Save segmentation results
    # Load one model at a time for testing.
    model_path = os.path.join(weights_dir, f"mitochondria_{fname}_50epochs_B_focal.hdf5")

if __name__ == '__main__':
    unet_model = UNet(input_shape)
    unetpp_model = UNetPP(input_shape)
    #funet_model = FUnet(input_shape)
    #att_unet_model = Attention_UNet(input_shape)
    #att_res_unet_model = Attention_ResUNet(input_shape)
    #models = [unet_model, att_unet_model, att_res_unet_model]
    models = [unetpp_model]
    #names = ['UNet', 'Attention UNet', 'Attention ResUNet']
    names = ['Unetplusplus']
    for model, name in zip(models, names):
        train_model(model,
                    optimizer=Adam(learning_rate=1e-2),
                    loss=BinaryFocalLoss(gamma=2),
                    metrics=['accuracy', dice_coef, jacard_coef],
                    epochs=50,
                    model_name=name)