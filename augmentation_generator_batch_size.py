import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import Sequence

class AugmentedCIFAR10Generator(Sequence):
    '''
    Class for data preparation for ResNet model with custom Augmentation class passed as an argument
    Works in two modes:
    1. With augmentor: applies augmentation upfront (more efficient for training)
    2. Without augmentor: works in batch-by-batch mode (better for validation/test)
    '''
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, augmentor=None, 
                 augment_fraction=0.2, num_classes=10, overwrite=False):
        self.x = x_data
        self.y = y_data.flatten()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentor = augmentor
        self.augment_fraction = augment_fraction
        self.overwrite = overwrite
        
        # If we have an augmentor, apply augmentation upfront (training mode)
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
            # No augmentor (test/validation mode)
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
            label_one_hot = np.zeros(self.num_classes, dtype=np.float32)
            label_one_hot[label] = 1.0
            
            img_pil = Image.fromarray(img)
            img_aug, label_aug = self.augmentor(img_pil, label_one_hot.copy())
            
            x_augmented.append(np.array(img_aug))
            y_augmented.append(np.argmax(label_aug))
            
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
            
            label_one_hot = np.zeros(self.num_classes, dtype=np.float32)
            label_one_hot[label] = 1.0
            
            batch_x.append(img)
            batch_y.append(label_one_hot)
            
        batch_x_resized = np.array([cv2.resize(np.array(img) if isinstance(img, Image.Image) 
                                               else img, (224, 224)) for img in batch_x])
        batch_x_preprocessed = preprocess_input(batch_x_resized.astype(np.float32))
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x_preprocessed, batch_y