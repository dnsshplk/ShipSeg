import tensorflow as tf

import numpy as np
import pandas as pd
import cv2
import os

import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

img_ow = img_oh = config['data']['img_orig_size']


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode provided by the Kaggle staff
def rle_decode(mask_rle, shape=(img_ow, img_oh)):
    if mask_rle == 0:
        return np.zeros(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  


def rle_encode(img, dshape=(img_ow, img_oh)):
    img = img.astype('float32')
    img = cv2.resize(img, dshape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Stack all masks on a single mask
def masks_as_image(in_mask_list):
    all_masks = np.zeros((img_ow, img_oh), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


# dsize = desired size
img_dw = img_dh = config['data']['img_dsize']

# We use one datagen class that can return both training and testing data
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images_dir, masks_csv, batch_size=16, image_size=(img_dw, img_dh), train = True, train_size = 0.8, no_ship_frac = 0.1):
        """
        imges_dir - folder with images
        masks_csv - df with ImageIds and encoded masks
        train - whether return training or testing data
        train_size - fraction of training data
        no_ship_frac - which fraction of no-ship images include in the dateset 
        """
        self.images_dir = images_dir
        self.masks_csv = pd.read_csv(masks_csv).fillna(0)

        # To avoid data leakage, use same random_state for .sample methods when initializing train and test datagens 
        self.balanced_masks_csv = pd.concat(
            [
                self.masks_csv[self.masks_csv['EncodedPixels'] == 0].sample(frac=no_ship_frac, random_state=42),
                self.masks_csv[self.masks_csv['EncodedPixels'] != 0]
            ]).sample(frac = 1, random_state=42)
        self.train = train
        self.train_size = train_size
        
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.all_batches = list(self.balanced_masks_csv.groupby('ImageId'))
        
        if self.train:
            self.all_batches = self.all_batches[:int(len(self.all_batches) * self.train_size)]
        else:
            self.all_batches = self.all_batches[int(len(self.all_batches) * self.train_size):]
        
        np.random.shuffle(self.all_batches)
        
    def __len__(self):
        return int(len(self.all_batches) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_images = self.all_batches[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        X = []
        y = []
        for batch_image in batch_images:
            image_path = os.path.join(self.images_dir, batch_image[0])
            
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_size)
            
            # Same preprocessing as in original ImageNet
            image = image.astype(np.float32) / 255.0
            image -= np.array([0.485, 0.456, 0.406])
            image /= np.array([0.229, 0.224, 0.225])

            mask = masks_as_image(batch_image[1]['EncodedPixels'])
            mask = cv2.resize(mask, self.image_size)
            mask = np.expand_dims(mask, axis=-1)
            
            # Thresholding the mask
            mask = mask > 0.5
            
            X.append(image)
            y.append(mask)
        
        return np.array(X), np.array(y).astype(np.float32)
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.all_batches)


