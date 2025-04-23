import cv2
import random, os
import numpy as np
import torch
from custom_transformers import *
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import Sequence

class AugmentedCIFAR10Generator(Sequence):
    '''
    Class for data preparation for RestNet model with custom Agumentation class passed as an argument
    '''
    def __init__(self, x_data, y_data, batch_size = 32, shuffle=True, augmentor = None, augment_fraction=0.2, num_classes =10):
        self.x = x_data
        self.y = y_data.flatten()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.augment_fraction = augment_fraction
        
        self.indices = np.arange(len(self.x))
        self.augmented_indices = np.random.choice(self.indices, size = int(len(self.x)* augment_fraction), replace=False) #choosing images to be augmented

    def __len__(self):
        return int(np.ceil(len(self.x)/ self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        batch_indices = self.indices[ index * self.batch_size : (index+1)* self.batch_size] #choosing batch at index 
        batch_x =[]
        batch_y=[]
        
        for i in batch_indices:
            img = self.x[i]
            label = self.y[i]
            label_one_hot = np.zeros(self.num_classes, dtype=np.float32)
            label_one_hot[label] = 1.0
            

            if self.augmentor and i in self.augmented_indices: #apply cutout
                img_pil = Image.fromarray(img)
                img_aug, label_aug = self.augmentor(img_pil, label_one_hot.copy())
                batch_x.append(img_aug)
                batch_y.append(label_aug)
            else:
                batch_x.append(img)
                batch_y.append(label_one_hot)
        batch_x_resized = np.array([
            cv2.resize(np.array(img) if isinstance(img, Image.Image) else img, (224, 224))
            for img in batch_x
        ])
        batch_x_preprocessed = preprocess_input(batch_x_resized.astype(np.float32))
        batch_y = np.array(batch_y, dtype=np.float32)  # important for soft labels

        return batch_x_preprocessed, batch_y
