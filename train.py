from utils.data import DataGenerator
from utils.evaluation import DiceBCELoss, IoU, DiceScore

from unet.model_build import get_unet_model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import yaml
import os
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)



def train_model(model, optimizer, loss, metrics, train_datagen, test_datagen, n_epochs):
   model.compile(optimizer=optimizer, loss=loss, metrics = metrics)

   history = model.fit(train_datagen,
                    epochs=n_epochs,
                    validation_data=test_datagen,
                    verbose = 1)
   
   return history, model


if __name__ == '__main__':
    # 1. Initialize model
    model = get_unet_model(output_channels=1)

    initial_learning_rate = config['train']['lr_scheduler']['initial_learning_rate']
    decay_steps = config['train']['lr_scheduler']['decay_steps']
    end_learning_rate = config['train']['lr_scheduler']['end_learning_rate']
    power = config['train']['lr_scheduler']['power']

    # 2. Initialize optimizer and lr scheduler
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate,
        decay_steps,
        end_learning_rate,
        power
    )

    optimizer = Adam(learning_rate = lr_schedule)

    # 3. Initialize loss function and metrics
    loss = DiceBCELoss
    metrics = [IoU, DiceScore]

    # 4. Initialize test and train DataGenerators
    img_dir = os.path.join('data', 'train_v2')
    mask_csv_path = os.path.join('data', 'train_ship_segmentations_v2.csv')

    train_datagen = DataGenerator(img_dir, mask_csv_path, train = True)
    test_datagen = DataGenerator(img_dir, mask_csv_path, train = False)

    # 5. Get the number of epochs from config
    n_epochs = config['train']['n_epochs']

    # 6. Begin training
    _, model = train_model(
        model = model,
        optimizer = optimizer,
        loss = loss,
        metrics = metrics,
        train_datagen = train_datagen,
        test_datagen = test_datagen,
        n_epochs = n_epochs
    )

    # 7. Save the model
    save_model = True

    if save_model:
        save_model_path = os.path.join('unet', 'custom_unet_last_weights.h5')
        model.save_weights(save_model_path)



