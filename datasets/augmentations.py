from torchvision import transforms
import random
from PIL import ImageOps, ImageFilter
import torch
    
class Resizer(object):

    def __init__(self, cfg):
        output_biggest_size = cfg.input_size
        input_width = cfg.width
        input_height = cfg.height
        biggest_side = min(input_width, input_height)
        
        self.scale = output_biggest_size/biggest_side
        self.output_width = input_width*self.scale
        self.output_height = input_height*self.scale

    def __call__(self, sample):
        annots = sample['annot']

        sample['image'] = transforms.functional.resize(img = sample['image'],
                                                       size = (int(self.output_height),
                                                               int(self.output_width)))
        
        annots[:, :4] *= self.scale
        sample['annot'] = annots
        sample['scale'] = self.scale
        return sample

class CenterCrop(object):

    def __init__(self, cfg):
        output_biggest_size = cfg.input_size
        input_width = cfg.width
        input_height = cfg.height
        biggest_side = min(input_width, input_height)
        
        scale = output_biggest_size/biggest_side
        input_width = input_width*scale
        input_height = input_height*scale
        
        self.size = cfg.center_crop_size
        self.offsets = ((input_height-cfg.center_crop_size)/2,
                        (input_width-cfg.center_crop_size)/2)
        
    def __call__(self, sample):
        annots = sample['annot']

        sample['image'] = transforms.functional.center_crop(img = sample['image'],
                                                            output_size  = (int(self.size)))
        
        annots[:,(0,2)] = annots[:,(0,2)] - self.offsets[1]
        annots[:,(1,3)] = annots[:,(1,3)] - self.offsets[0]
        sample['annot'] = annots
        return sample

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        
        if torch.rand(1)[0] < self.p:
            sigma = random.random() * 1.9 + 0.1
            x["image"] = x["image"].filter(ImageFilter.GaussianBlur(sigma))
        else:
            pass
        
        return x

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        
        if torch.rand(1)[0] < self.p:
            x["image"] = ImageOps.solarize(x["image"])
        else:
            pass
        
        return x

class RandomVerticalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1)[0] < self.p:
            sample['image'] = transforms.functional.vflip(sample['image'])

            annots = sample['annot']
            channels, rows, cols = sample['image'].shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample['annot'] = annots

        return sample
    
class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):

        if torch.rand(1)[0] < self.p:
            sample['image'] = transforms.functional.hflip(sample['image'])

            annots = sample['annot']
            channels, rows, cols = sample['image'].shape
            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()
            y_tmp = y1.copy()
            annots[:, 1] = rows - y2
            annots[:, 3] = rows - y_tmp
            sample['annot'] = annots

        return sample

class ColorJitter(object):
    def __init__(self,
                 brightness=0.4,
                 contrast=0.4,
                 saturation=0.2,
                 hue=0.1, p=0.2):
        
        self.p = p
        self.apply = transforms.ColorJitter(brightness=0.4,
                                            contrast=0.4,
                                            saturation=0.2,
                                            hue=0.1)

    def __call__(self, x):
        
        if random.random() < self.p:
            x["image"] = self.apply(x["image"])
        else:
            pass
        
        return x
    
class RandomGrayscale(object):
    def __init__(self,
                 brightness=0.4,
                 contrast=0.4,
                 saturation=0.2,
                 hue=0.1, p=0.2):
        
        self.p = p
        self.apply = transforms.RandomGrayscale()

    def __call__(self, x):
        
        if random.random() < self.p:
            x["image"] = self.apply(x["image"])
        else:
            pass
        
        return x

class ToTensor(object):
    def __init__(self):
        self.apply = transforms.ToTensor()
    
    def __call__(self, x):
        x["image"] = self.apply(x["image"])
        return x
    
class Normalize(object):
    def __init__(self,mean = [0.485, 0.456, 0.406],
                      std  = [0.229, 0.224, 0.225]):
        self.apply = transforms.Normalize(mean, std)
    
    def __call__(self, x):
        x["image"] = self.apply(x["image"])
        return x

class Identity(object):
    def __call__(self, x):
        return x

class TransformTrain:
    
    # RandomVerticalFlip: Done
    # RandomHorizontalFlip: Done
    # ColorJitter: Done
    # RandomGrayscale: Done
    # GaussianBlur: Done
    # Solarization: Done
    # ToTensor: Done
    # Normalize: Done
    # Identity: Done
    
    def __init__(self, cfg):
        self.transform = transforms.Compose([
            ColorJitter(brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1, p=0.2) if cfg.augmentations.color_jitter else Identity(),
            RandomGrayscale(p=0.1) if cfg.augmentations.random_gray_scale else Identity(),
            GaussianBlur(p=0.1) if cfg.augmentations.gaussian_blur else Identity(),
            Solarization(p=0.1) if cfg.augmentations.solarization else Identity(),
            Resizer(cfg),
            CenterCrop(cfg),
            ToTensor(),
            RandomVerticalFlip(p=0.2) \
                if cfg.augmentations.random_vertical_flip else Identity(),
            RandomHorizontalFlip(p=0.2) \
                if cfg.augmentations.random_horizontal_flip else Identity(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]) \
                if cfg.augmentations.normalize else Identity()
            ])

    def __call__(self, x):
        y = self.transform(x)
        return y
    
class TransformEvaluate:
    def __init__(self, cfg):
        self.transform = transforms.Compose([Resizer(cfg),
                                             CenterCrop(cfg),
                                             ToTensor(),
                                             Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]) if cfg.augmentations.normalize else Identity()])
    def __call__(self, x):
        y = self.transform(x)
        return y
    
