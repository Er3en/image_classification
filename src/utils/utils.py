import numpy as np
import progressbar 
from progressbar import *
from rich.progress import track

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