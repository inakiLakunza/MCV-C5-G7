import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from PIL import Image

import shutil

#from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ORIGINAL_DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
NEW_DATASET_DIR = '/ghome/group07/MIT_small_augments/small_bigger_3'

if not os.path.exists(NEW_DATASET_DIR): 
    os.makedirs(NEW_DATASET_DIR)
    print(f'Directory created: {NEW_DATASET_DIR}')
else: print(f'Directory was already created: {NEW_DATASET_DIR}')


IMG_SIZE = 256
BATCH_SIZE=32

class_names=['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding'],
  

train_data_uploader = ImageDataGenerator(
    #rotation_range=1.30,
    #width_shift_range=0.002,
    #height_shift_range=0.49,
    #shear_range=0.43,
    #zoom_range=0.15,
    #horizontal_flip=True,
    vertical_flip=False
)

# Load and preprocess the training dataset
train_dataset = train_data_uploader.flow_from_directory(
    directory=ORIGINAL_DATASET_DIR+'/train/',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

generator1 = ImageDataGenerator(
    rotation_range=1.30,
    #width_shift_range=0.002,
    #height_shift_range=0.49,
    #shear_range=0.43,
    #zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False
)


'''
i = 0
for batch in train_data_uploader.flow_from_directory(ORIGINAL_DATASET_DIR+'/train', target_size=(256,256),
    class_mode='categorical'
    shuffle=False, batch_size=1,
    save_to_dir='/ghome/group07/new_databases/prueba1/train/coast', save_prefix='copy_', save_format='jpg'):

    i += 1
    if i > 5: # save 20 images
        break  # otherwise the generator would loop indefinitely
'''

'''
for batch in generator1.flow_from_directory(ORIGINAL_DATASET_DIR+'/train',
                    target_size=(256,256),
                    class_mode='categorical',
                    batch_size=1,
                    shuffle=False,
                    save_to_dir=new_to_save,
                    save_format="jpg",
                    save_prefix=''
    ):
    break
'''

generator2 = ImageDataGenerator(
    #rotation_range=1.30,
    #width_shift_range=0.002,
    #height_shift_range=0.49,
    #shear_range=0.43,
    #zoom_range=0.15,
    #horizontal_flip=True,
    #vertical_flip=False
)


old_label = "a"
i=0
sum=0

for img, label in train_dataset:

    label_name = class_names[0][np.argmax(label)]
    if label_name != old_label: i=0

    img1=img[0]
    save_dir = os.path.join(NEW_DATASET_DIR, 'train', label_name) 

    '''
    ori_name = 'copy'+str(i)+'.jpg'
    im = Image.fromarray((img1).astype(np.uint8))
    ori_to_save = os.path.join(save_dir, ori_name) 
    im.save(ori_to_save)
    '''
    # AUGMENT 1
    tr_params1 = {'theta':  15, 
                 #'tx'   : 0.1,
                 #'ty'   : 0.1,
                 'zx'   : 0.9,
                 'zy'   : 0.9,
                 'flip_horizontal':True,
                 'brightness': 0.9}

    new_img1 = generator2.apply_transform(img1, tr_params1)

    new_im1 = Image.fromarray((new_img1).astype(np.uint8))
    new_name1 = 'mod1_'+str(i)+'.jpg'
    new_to_save1 = os.path.join(save_dir, new_name1) 
    new_im1.save(new_to_save1)

    
    # AUGMENT 2
    tr_params2 = {'theta':  -15, 
                 'tx'   : 0.1,
                 'ty'   : 0.1,
                 'zx'   : 0.95,
                 'zy'   : 0.95,
                 'flip_horizontal':True,
                 'brightness': 1.05}


    new_img2 = generator2.apply_transform(img1, tr_params2)

    new_im2 = Image.fromarray((new_img2).astype(np.uint8))
    new_name2 = 'mod2_'+str(i)+'.jpg'
    new_to_save2 = os.path.join(save_dir, new_name2) 
    new_im2.save(new_to_save2)

    # AUGMENT 3
    tr_params3 = {'theta': 25, 
                 'tx'   : -0.1,
                 'ty'   : -0.1,
                 'zx'   : 0.95,
                 'zy'   : 0.95,
                 'flip_horizontal':False,
                 'brightness': 1.1}


    new_img3 = generator2.apply_transform(img1, tr_params3)

    new_im3 = Image.fromarray((new_img3).astype(np.uint8))
    new_name3 = 'mod3_'+str(i)+'.jpg'
    new_to_save3 = os.path.join(save_dir, new_name3) 
    new_im3.save(new_to_save3)
    

    if i%50==0: print(f'{label_name}   {i}')

    i+=1
    old_label = label_name
    sum+=1
    if sum==400:
        print('Task finished')
        break
