import os
import sys
import argparse
import pandas as pd
import segmentation_models as sm
from data import DataGenerator
from callbacks.history import History

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import MeanIoU, IoU

if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM environment variables to determine training and cross validation set number
    slurm_array_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))
    slurm_array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")

    train_file = "{:02d}".format(slurm_array_task_id)
    cross_file = "{:02d}".format(slurm_array_task_id)
    print(f"Train file: {train_file}")
    print(f"Cross file: {cross_file}")

    # Configure directories
    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, 'data', 'projects', 'consorcio-ia', 'data', 'oil_spills_17', 'augmented_dataset')
    feat_dir = os.path.join(data_dir, 'features')
    labl_dir = os.path.join(data_dir, 'labels')
    train_dir = os.path.join(data_dir, 'learningCSV', 'trainingFiles')
    cross_dir = os.path.join(data_dir, 'learningCSV', 'crossFiles')

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)

    # Features to use
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

    # Load CSV key files
    train_set = pd.read_csv(os.path.join(train_dir, f"train{train_file}.csv"))
    cross_set = pd.read_csv(os.path.join(cross_dir, f"cross{cross_file}.csv"))
    print(f"Training CSV file length: {len(train_set)}")
    print(f"Cross validation CSV file length: {len(cross_set)}")

    # Load generators
    train_keys = train_set["key"]
    cross_keys = cross_set["key"]
    train_gen = DataGenerator(keys = train_keys, 
                              features_path = feat_dir, 
                              labels_path = labl_dir, 
                              features_channels = feat_channels, 
                              feature_ext = '.tiff', 
                              label_ext = '.pgm', 
                              batch_size = 64,
                              dimensions = [224, 224, 3])
    cross_gen = DataGenerator(keys = cross_keys,
                              features_path = feat_dir,
                              labels_path = labl_dir,
                              features_channels = feat_channels,
                              feature_ext = '.tiff',
                              label_ext = '.pgm',
                              batch_size = 32,
                              dimensions = [224, 224, 3])
    print(f"Training generator set length: {len(train_gen)}")
    print(f"Cross validation generator set length: {len(cross_gen)}")
    
    #
    # Load and configure model (optimizer, loss functions, callbacks)

    # Callbacks
    # checkpoint = ModelCheckpoint(os.path.join(args.results_path, f"unet_checkpoint_set{train_file}_epochs{args.num_epochs}.keras"), monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')
    # history = History(os.path.join(args.results_path, f"history_{train_file}"),
    #                  os.path.join(args.results_path, f"loss_{train_file}"),
    #                  10, 
    #                  "accuracy")
    logger = CSVLogger(os.path.join(args.results_path, f"unet_training_set{train_file}_epochs{args.num_epochs}.log"))
    callbacks = [logger]
    # Optimizer
    optimizer = Adam(learning_rate = 1e-3)
    # Model
    model = sm.Unet('resnet34', classes=1, activation='sigmoid')
    # Metrics
    meaniou = MeanIoU(num_classes=2)
    iou_c0 = IoU(num_classes=2, target_class_ids=[0], name='iou_class0')
    iou_c1 = IoU(num_classes=2, target_class_ids=[1], name='iou_class1')
    # Compile with loss function
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy','precision','recall',meaniou,iou_c0,iou_c1])
    model.summary()
    
    # Train
    model.fit(train_gen, 
              validation_data = cross_gen,
              epochs = int(args.num_epochs),
              callbacks = callbacks,
              verbose = 1)
    # history.savePlot()
    # history.saveCSV()
    #
    #
    # Test
    #
    #
    #
    # Save params
    #
    #
    # Save results
    print("Done!")
