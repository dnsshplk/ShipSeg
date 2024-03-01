import matplotlib.pyplot as plt
import numpy as np


# PLot prediction of a batch (NOT USED)
def plot_img_mask_pred(batch, model):
    Xs = batch[0]
    ys = batch[1]

    batch_size = Xs.shape[0]

    fig, axs = plt.subplots(batch_size, 3, figsize = (10 * 3, 10 * batch_size))

    for i in range(batch_size):

        axs[i, 0].imshow(Xs[i] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        pred_y = model(np.expand_dims(Xs[i], 0))

        axs[i, 1].imshow(ys[i])
        axs[i, 2].imshow(pred_y.numpy().squeeze())

    plt.show()