import tensorflow as tf
import yaml

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

img_w = img_h = config['data']['img_dsize']


# Using pretrained ResNet50 as an encoder
base_model = tf.keras.applications.resnet50.ResNet50(include_top = False, weights='imagenet', input_shape = [img_w, img_w, 3])

# Names of a layers, whose outputs will be connected with an decoder
layer_names = [
    'conv1_relu',
    'conv2_block3_out', 
    'conv3_block4_out',  
    'conv4_block6_out',  
    'conv5_block3_out',  
]

# Skip-connection-layer outputs
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Creating an encoder
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = True


# Custom upsample block of a decoder
def upsample(filters, size):

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same'))

  result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.ReLU())

  return result

# List of all upsample blocks of the decoder
up_stack = [
    upsample(512, 3),
    upsample(256, 3),
    upsample(128, 3),
    upsample(64, 3),
]


# Function that will build and return U-Net
def get_unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[img_w, img_w, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same') 
    
    x = last(x)
    
    x = tf.keras.activations.sigmoid(x)

    return tf.keras.Model(inputs=inputs, outputs=x)