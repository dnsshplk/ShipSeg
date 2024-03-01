import tensorflow as tf

from utils.data import DataGenerator
from utils.evaluation import DiceBCELoss, IoU, DiceScore

from unet.model_build import get_unet_model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import yaml
import os

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


if __name__ == '__main__':
    weights_path = os.path.join('unet', 'custom_unet_weights.h5')

    output_channels = config['data']['n_classes']

    model = get_unet_model(output_channels=output_channels)

    model.load_weights(weights_path)

    optimizer = Adam(learning_rate = 0.001)

    model.compile(optimizer, loss=DiceBCELoss, metrics = [IoU, DiceScore])

    img_dir = os.path.join('data', 'train_v2')
    mask_csv_path = os.path.join('data', 'train_ship_segmentations_v2.csv')

    # Evaluate the model on imbalanced (original) data: no_ship_frac=1
    eval_data = DataGenerator(img_dir, mask_csv_path, train = False, no_ship_frac=1)

    model.evaluate(eval_data)