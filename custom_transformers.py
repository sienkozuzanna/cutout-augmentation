import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

class RandomPixelCutout:
    def __init__(self, max_cutout_size):
        '''
        Class performing random pixels cutout and returning new Image object after transformation and new soft label.
        :params  max_cutout_size: maximm percent value of pixels that can be cut out.
        '''
        self.max_cutout_size = max_cutout_size

    def __call__(self, img, label):
        '''
        Method for performing transformations.
        : params img : Image object 
        : param label : original tabel for the provided image. Should be 0-1.
        returns new Image with a new soft label based on % of remaining original pixels
        '''
        img = np.array(img)
        h,w,_ = img.shape
        total_pixels = h*w
        # randomly choosing number of pixels to cover
        num_to_remove = random.randint(1, int(self.max_cutout_size * total_pixels))

        # randomly choosing x,y coordinates of pixels that will be cutout
        y_coords = np.random.randint(0, h, size=num_to_remove)
        x_coords = np.random.randint(0, w, size=num_to_remove)

        img[y_coords, x_coords, :] = 0 #change pixels to black

        #calculate new label based on remaining fraction
        remaining = total_pixels - num_to_remove
        new_label = label* remaining/total_pixels
        return Image.fromarray(img), new_label
    
class SquareCutout:
    
    def __init__(self, size):
        '''
        Class performing square cutout and returning new Image object after transformation and new soft label.
        :params  size: size of the square cutout in pixels.
        '''
        self.size = size

    def __call__(self, img, label):
        '''
        Method for performing transformations.
        : params img : Image object 
        : param label : original tabel for the provided image. Should be 0-1.
        returns new Image with a new soft label based on % of remaining original pixels
        '''
        img = np.array(img)
        h,w,_ = img.shape
        # ensuring that the cutout size is smaller than the image dimensions
        # if the cutout size is larger than the image dimensions, raise an error
        if self.size > min(h,w):
            raise ValueError(f"Cutout size {self.size} is larger than image dimensions ({h}, {w})")
        # randomly choosing x,y coordinates of the top left corner of the square cutout
        x = np.random.randint(0, w - self.size)
        y = np.random.randint(0, h - self.size)

        img[y:y+self.size, x:x+self.size, :] = 0 #change pixels to black

        #calculate new label based on remaining fraction
        total_pixels = h*w
        num_to_remove = self.size**2
        remaining = total_pixels - num_to_remove
        new_label = label* remaining/total_pixels
        return Image.fromarray(img), new_label
    

class ColorCutout:
    def __init__(self, size, color=None):
        '''
        Class performing color square cutout and returning new Image object after transformation and new soft label.
        :params  size: size of the square cutout in pixels.
        :params color: color of the square cutout in RGB format. (if not provided - randomly chossen from dictionary)
        '''
        self.size = size
        # chceking if the color is a tuple of 3 integers representing RGB values
        # and if the values are between 0 and 255
        if color is not None:
            if not isinstance(color, tuple) or len(color) != 3:
                raise ValueError("Color must be a tuple of 3 integers representing RGB values.")
            if any(c < 0 or c > 255 for c in color):
                raise ValueError("RGB values must be between 0 and 255.")
        # dictionary of colors to choose from
        else:
            # colors
            colors = {
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'purple': (128, 0, 128),
                'cyan': (0, 255, 255),
                'orange': (255, 165, 0),
                'pink': (255, 192, 203),
                'beige': (245, 222, 179),
                }
            # randomly choosing color from the dictionary
            color = random.choice(list(colors.values()))
        self.color = color


    def __call__(self, img, label):
        '''
        Method for performing transformations.
        : params img : Image object 
        : param label : original tabel for the provided image. Should be 0-1.
        returns new Image with a new soft label based on % of remaining original pixels
        '''

        
        img = np.array(img)
        h,w,_ = img.shape
        # ensuring that the cutout size is smaller than the image dimensions
        # if the cutout size is larger than the image dimensions, raise an error
        if self.size > min(h,w):
            raise ValueError(f"Cutout size {self.size} is larger than image dimensions ({h}, {w})")
        # randomly choosing x,y coordinates of the top left corner of the square cutout
        x = np.random.randint(0, w - self.size)
        y = np.random.randint(0, h - self.size)

    
        # creating a square cutout of the specified color
        img[y:y+self.size, x:x+self.size, :] = self.color

        #calculate new label based on remaining fraction
        total_pixels = h*w
        num_to_remove = self.size**2
        remaining = total_pixels - num_to_remove
        new_label = label* remaining/total_pixels
        return Image.fromarray(img), new_label
        

class SoftLabelDataset(Dataset):
    
    def __init__(self, dataset, pipeline_before_cutout, pipeline_after_cutout, num_classes, cutout_transform=None):
        '''
        Class for performing transformations and adding new rows to the dataset.
        :params dataset : dataset to be transformed
        :params pipeline_before_cutout : pipeline for transformations to be done before performing cutout.
        :params pipeline_after_cutout : pipeline for transformations to be done after performing cutout.
        :params num_classes : number of classes in the dataset. Used for One-Hot Ecoding.
        :params cutout_transform : transformer object performing cutout to be used. If None - only before and after cutout transformations will be performed and no data will be added.
        '''
        self.dataset = dataset
        self.pipeline_before_cutout = pipeline_before_cutout
        self.pipeline_after_cutout = pipeline_after_cutout
        self.cutout_transform = cutout_transform
        self.num_classes = num_classes

    def __len__(self):
        # If cutout will be performed dataset length will be 2x greater
        return len(self.dataset) * 2 if self.cutout_transform is not None else len(self.dataset)
    
    def __getitem__(self, idx):
        if self.cutout_transform is not None:
            orig_idx = idx // 2
            is_augmented = idx % 2 == 1
        else:
            orig_idx = idx
            is_augmented = False

        # Original image and label + one hot encoding
        image, label = self.dataset[orig_idx]
        label = F.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).float()
        
        # Before cutout transformations regardless of the cutout choice
        pil_img = self.pipeline_before_cutout(image)

        # Performing cutout only if there is a cutout transformer, else just after_cutout transformations
        if is_augmented:
            cutout_img, new_label = self.cutout_transform(pil_img, label)
            return self.pipeline_after_cutout(cutout_img), new_label
        else:
            return self.pipeline_after_cutout(pil_img), label
