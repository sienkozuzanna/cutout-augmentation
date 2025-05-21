import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

class AugmentedCIFAR10Generator(Sequence):
    '''
    Class for data preparation for ResNet model with custom Augmentation class passed as an argument
    Works in two modes:
    1. With augmentor: applies augmentation upfront (more efficient for training)
    2. Without augmentor: works in batch-by-batch mode (better for validation/test)
    Preserves augmented labels exactly as returned by the augmentor
    '''
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, augmentor=None, 
                 augment_fraction=0.2, num_classes=10, overwrite=False, soft_label = True, soft_label_fraction = 0):
        self.x = x_data
        self.y = y_data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.augment_fraction = augment_fraction
        self.overwrite = overwrite
        self.soft_label = soft_label
        self.soft_label_fraction = soft_label_fraction
        
        if len(self.y.shape) == 1 or self.y.shape[1] != self.num_classes:
            #self.y = np.eye(self.num_classes)[self.y.flatten()]
            self.y = to_categorical(self.y, num_classes=self.num_classes)
        
        if self.augmentor is not None:
            self.x_augmented, self.y_augmented = self._apply_augmentation()
            
            if not self.overwrite:
                self.x_combined = np.concatenate([self.x, self.x_augmented])
                self.y_combined = np.concatenate([self.y, self.y_augmented])
            else:
                self.x_combined = self.x.copy()
                self.y_combined = self.y.copy()
                self.x_combined[self.augmented_indices] = self.x_augmented
                self.y_combined[self.augmented_indices] = self.y_augmented
        else:
            self.x_combined = self.x
            self.y_combined = self.y
        
        self.indices = np.arange(len(self.x_combined))
        self.on_epoch_end()

    def _apply_augmentation(self):
        self.augmented_indices = np.random.choice(
            np.arange(len(self.x)), 
            size=int(len(self.x) * self.augment_fraction), 
            replace=False
        )
        
        x_augmented = []
        y_augmented = []
        
        for i in self.augmented_indices:
            img = self.x[i]
            label = self.y[i]
            
            img_pil = Image.fromarray(img)
            img_aug, label_aug = self.augmentor(img_pil, label.copy())

            # convert the soft label so that it's not dependent on the area
            # (only if soft_label_fraction specified)

            if self.soft_label_fraction:
                label_aug = [0 if k==0 else self.soft_label_fraction for k in label_aug]
            
            x_augmented.append(np.array(img_aug))
            if self.soft_label:
                y_augmented.append(label_aug)
            else:
                y_augmented.append(label) #if we dont want to change labels
            
        return np.array(x_augmented), np.array(y_augmented)

    def __len__(self):
        return int(np.ceil(len(self.x_combined) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for i in batch_indices:
            img = self.x_combined[i]
            label = self.y_combined[i]
            
            batch_x.append(img)
            batch_y.append(label)
            
        # batch_x_resized = np.array([cv2.resize(np.array(img) if isinstance(img, Image.Image) 
        #                                        else img, (224, 224)) for img in batch_x])
        batch_x_normalized = np.array(batch_x, dtype = np.float32)/255.0
        #batch_x_preprocessed = preprocess_input(batch_x_resized.astype(np.float32))
        batch_y = np.array(batch_y, dtype=np.float32) 

        return batch_x_normalized, batch_y
