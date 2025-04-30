import os
import matplotlib.pyplot as plt

def plot_history(model_history, plots_dir, fname):
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
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{fname}_accuracy.png"))
    plt.close()

