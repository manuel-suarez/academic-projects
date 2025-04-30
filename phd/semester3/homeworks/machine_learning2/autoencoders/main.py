ComputeLB = False

import os, gc, zipfile
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

if ComputeLB: PATH = '../input/generative-dog-images/all-dogs/all-dogs/'
else: PATH = '/home/est_posgrado_manuel.suarez/data/all-dogs'
IMAGES = os.listdir(PATH)
print('There are',len(IMAGES),'images. Here are 5 example filesnames:')
print(IMAGES[:5])

#os.mkdir('../tmp')
#os.mkdir('../tmp/images')

# CREATE RANDOMLY CROPPED IMAGES
for i in range(500000):
    img = Image.open(os.path.join(PATH, IMAGES[i%len(IMAGES)]))
    img = img.resize(( 100,int(img.size[1]/(img.size[0]/100) )), Image.ANTIALIAS)
    w = img.size[0]; h = img.size[1]; a=0; b=0
    if w>64: a = np.random.randint(0,w-64)
    if h>64: b = np.random.randint(0,h-64)
    img = img.crop((a, b, 64+a, 64+b))
    img.save('../tmp/images/'+str(i)+'.png','PNG')
    if i%100000==0: print('created',i,'cropped images')
print('created 500000 cropped images')

