import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_crossentropy


def DiceScore(targets, inputs, smooth=1):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice


def DiceBCELoss(targets, inputs, smooth=1e-6):    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.sum(targets * inputs)    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE


def IoU(targets, inputs, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU