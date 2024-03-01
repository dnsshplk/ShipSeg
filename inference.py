import tensorflow as tf
from unet.model_build import get_unet_model
from utils.data import rle_encode
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


def predict(model, input_folder, output_folder, save_masks = False, visualize = False):
    """
    model - model for inference
    input_folder - folder with images
    output_folder - folder where masks and res_df are saved
    save_masks - whether to save masks
    visualize - in case there is not many images, visualize the predictions
    """

    images_list = os.listdir(input_folder)

    if len(images_list) > 8:
        visualize = False
        print('Too many images: visualize = False')
        

    # Encoded rle masks. In the end this list will be a column in res_df
    mask_rle_list = []

    image_size = (config['data']['img_dsize'], config['data']['img_dsize'])

    if visualize:
        fig, axs = plt.subplots(len(images_list), 2, figsize = (10 * 2, 10*len(images_list)))

    # Load, preprocess, predict and save in a loop  
    for i, img_name in enumerate(images_list):
        image_path = os.path.join(input_folder, img_name)
            
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)
            
        image = image.astype(np.float32) / 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
            
        pred = model.predict(np.expand_dims(image, 0))

        pred = pred > 0.5
        
        if save_masks:
            mask_path = os.path.join(output_folder, img_name)

            cv2.imwrite(mask_path, pred.squeeze() * 255.0)

        mask_rle = rle_encode(pred.squeeze())

        mask_rle_list.append(mask_rle)

        if visualize:
            axs[i, 0].imshow(image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
            axs[i, 1].imshow(pred.squeeze())

    if visualize:
        plt.show()

    # Create a result df and save it
    res_df = pd.DataFrame({'index': list(images_list), 'masks':mask_rle_list})
    df_path = os.path.join(output_folder, 'res_df.csv')
    res_df.to_csv(df_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ship Segmentaion")

    parser.add_argument('-i', '--input', dest='input_folder', help='Path to the input folder', default='input_folder')
    parser.add_argument('-o', '--output', dest='output_folder', help='Path to the output folder', default='output_folder')
    parser.add_argument('-v', '--visualize', action='store_true', help='Enable verbose mode')
    parser.add_argument('-sm', '--savemask', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()


    input_folder = args.input_folder
    output_folder = args.output_folder
    save_masks = args.savemask
    visualize = args.visualize

    weights_path = os.path.join('unet', 'custom_unet_weights.h5')

    output_channels = config['data']['n_classes']

    model = get_unet_model(output_channels=output_channels)

    model.load_weights(weights_path)

    predict(model, input_folder, output_folder, save_masks=save_masks, visualize = visualize)



    