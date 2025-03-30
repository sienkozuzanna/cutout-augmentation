import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F

class RandomPixelCutout:
    def __init__(self, max_cutout_size, color=False):
        '''
        Class performing random pixels cutout and returning new Image object after transformation and new soft label.
        :params  max_cutout_size: maximm percent value of pixels that can be cut out.
        :params  color: if false - black, if True - random colors.
        '''
        self.max_cutout_size = max_cutout_size
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
        total_pixels = h*w
        # randomly choosing number of pixels to cover
        num_to_remove = random.randint(1, int(self.max_cutout_size * total_pixels))

        # randomly choosing x,y coordinates of pixels that will be cutout
        y_coords = np.random.randint(0, h, size=num_to_remove)
        x_coords = np.random.randint(0, w, size=num_to_remove)

        if self.color:
            # if color is provided - change pixels to selected color
            for i in range(num_to_remove):
                # generate random color
                r = np.random.randint(0, 256)
                g = np.random.randint(0, 256)
                b = np.random.randint(0, 256)
                img[y_coords[i], x_coords[i], :] = (r, g, b)
        else:
            img[y_coords, x_coords, :] = 0 #change pixels to black

        #calculate new label based on remaining fraction
        remaining = total_pixels - num_to_remove
        new_label = label* remaining/total_pixels
        return Image.fromarray(img), new_label
    
class SquareCutout:
    
    def __init__(self, size, color=False):
        '''
        Class performing square cutout and returning new Image object after transformation and new soft label.
        :params  size: size of the square cutout in pixels.
        :params  color: if fals e - black, if True - random colors.
        '''
        self.size = size
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
        if self.color:
            # if color is True - randomly generate color
            for i in range(x, x+self.size):
                for j in range(y, y+self.size):
                    # generate random color
                    r = np.random.randint(0, 256)
                    g = np.random.randint(0, 256)
                    b = np.random.randint(0, 256)
                    img[i, j, :] = (r,g,b)
        else:       
            img[y:y+self.size, x:x+self.size, :] = 0 #change pixels to black

        #calculate new label based on remaining fraction
        total_pixels = h*w
        num_to_remove = self.size**2
        remaining = total_pixels - num_to_remove
        new_label = label* remaining/total_pixels
        return Image.fromarray(img), new_label
    
class CircleCutout:
    def __init__(self, radius=None, max_size_ratio=0.3, color=None, random_color=False):
        '''
        Class performing circle cutout and returning new Image object after transformation and new soft label.
        :params  radius: radius of the circle cutout in pixels.
        :param max_size_ratio: maximum size of the circle relative to image dimensions (0-1)
        :params  color: if None - black cutout
        :params random_color: if True apply random colors in masked pixels
        '''

        self.radius=radius
        self.max_size_ratio = max_size_ratio
        self.color = color if color is not None else (0, 0, 0) #black cutout by default
        self.random_color=random_color

    def __call__(self, img, label):
        img = np.array(img)
        h,w,_ = img.shape
        max_possible_radius = int(min(h, w) * self.max_size_ratio / 2)

        if self.radius is None:
            radius = random.randint(5, max_possible_radius)
        else:
            radius = min(self.radius, max_possible_radius)
            if radius != self.radius:
                print(f"Warning: Reduced radius from {self.radius} to {radius} to fit image")

        #center of circle cutout and ensuring it fits in image size
        x_center = random.randint(radius, w - radius - 1) if w > 2*radius else w // 2
        y_center = random.randint(radius, h - radius - 1) if h > 2*radius else h // 2

        #open grid for generating circle coordinates
        y, x = np.ogrid[:h, :w]
        #generating circle mask of pixels from equation (x-x_center)^2+(y-y_center)^2<=radius^2
        mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2

        if self.random_color:
            # generating random colors only for masked pixels
            random_colors = np.random.randint(0, 256, size=(np.sum(mask), 3), dtype=np.uint8)
            img[mask] = random_colors
        else:
            img[mask] = self.color

        #adjusting label
        total_pixels = h * w
        num_removed = np.sum(mask)
        new_label = label * (1 - num_removed / total_pixels)
        
        return Image.fromarray(img), new_label
        
    
class PolygonCutout:
    def __init__(self, max_vertices=12, min_vertices=3, max_size_ratio=0.3, color=None, random_color=False):
        '''
        Class performing polygon cutout with configurable maximum number of vertices.
        
        :param max_vertices: maximum number of vertices for the polygon (default 12)
        :param max_size_ratio: maximum size of the polygon relative to image dimensions (0-1)
        :param min_vertices: minimum number of vertices (default 3 for triangles)
        :param color: RGB color for the cutout. Default is black.
        :param random_color: if True apply random colors to masked pixels
        '''

        self.max_vertices = max(max_vertices, min_vertices)
        self.min_vertices = max(min_vertices, 3)
        self.max_size_ratio = max_size_ratio
        self.color = color if color is not None else (0, 0, 0) #black cutout by default
        self.random_color=random_color

    def _is_inside_polygon(self, x, y, polygon):
        '''
        Ray-casting algorithm to check if a point (x, y) is inside the polygon.
        '''
        num_vertices = len(polygon)
        inside = False
        x0, y0 = polygon[-1]

        for i in range(num_vertices):
            x1, y1 = polygon[i]
            if ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / (y0 - y1) + x1):
                inside = not inside
            x0, y0 = x1, y1

        return inside

    def __call__(self, img, label):
        '''
        Method for performing transformations.
        : params img : Image object 
        : param label : original tabel for the provided image. Should be 0-1.
        returns new Image with a new soft label based on % of remaining original pixels
        '''
        img = np.array(img)
        h,w,_ = img.shape
        
        vertices=random.randint(self.min_vertices, self.max_vertices)
        max_size = int(min(h, w) * self.max_size_ratio) #maximum possible size of cutout
        center_x = random.randint(max_size, w - max_size) if w > 2*max_size else w // 2
        center_y = random.randint(max_size, h - max_size) if h > 2*max_size else h // 2

        points=[]
        angle_step=2*np.pi/vertices #angle between vertices in radial coordinates
        base_angle = random.uniform(0, 2 * np.pi) #random start point

        for i in range(vertices):
            #start point+theoretical point on circle + noise
            angle = base_angle + i * angle_step + random.uniform(-angle_step/3, angle_step/3)
            #every point has random distance between center and max size of cutout
            distance = random.uniform(max_size * 0.3, max_size * 0.7)

            #radial coordinates to cartesian coordinates
            x, y = center_x + distance * np.cos(angle), center_y + distance * np.sin(angle)
            #ensuring that we wont exceed image size
            points.append((max(0, min(w-1, x)), max(0, min(h-1, y))))

        num_removed = 0
        for y in range(h):
            for x in range(w):
                if self._is_inside_polygon(x, y, points):
                    if self.random_color:
                        # generating random color for each pixel
                        img[y, x, :] = (
                            np.random.randint(0, 256),
                            np.random.randint(0, 256),
                            np.random.randint(0, 256)
                        )
                    else:
                        img[y, x, :] = self.color
                    num_removed += 1

        total_pixels = h * w
        remaining = total_pixels - num_removed
        new_label = label * (remaining / total_pixels)

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
