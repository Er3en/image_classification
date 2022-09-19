import numpy as np
import progressbar 
from progressbar import *
from rich.progress import track
import matplotlib.pyplot as plt

def center_image(img):
    size = [256, 256]
    img_size = img.shape[:2]
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    return resized

def progress_bar(array):
    widgets = [' Progress: ', SimpleProgress(),
               ', Percent: ', Percentage(),
               ' ', ETA(),
               ' ', AdaptiveETA()]
    
    bar = progressbar.ProgressBar(maxval=len(array),widgets=widgets).start()
    return bar

def show_images(train_df, images, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))
    for i in range(10):
        path = train_df[train_df.category == i].values[2]
        axes[i//cols, i%cols].set_title(path[0].split('/')[-2] + str(path[1]))
        axes[i//cols, i%cols].imshow(images[train_df[train_df.filename == path[0]].index[0]])
    plt.show()