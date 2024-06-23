from torchvision.transforms import functional
from torchvision import transforms
import torch
import random
import numpy as np

class PairedRandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, cond_img=None):
        if random.random() < self.flip_prob:
            image = functional.hflip(image)

            if cond_img is not None:
                cond_img = functional.hflip(cond_img)

            target = functional.hflip(target)
        
        return image, target, cond_img


class TripletRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(
        self, 
        img, 
        depth,
        normal,
        mask
    ):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        
        if torch.rand(1) < self.p:
            img = transforms.functional.hflip(img)
            normal = transforms.functional.hflip(normal)        
            normal[0, :, :] = -normal[0, :, :]
            depth = transforms.functional.hflip(depth)
            mask = transforms.functional.hflip(mask)

        return img, depth, normal, mask

        
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, img, tgt, interpolation1=None, interpolation2=None, is_surface_normal=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
        # if True:
            img = transforms.functional.hflip(img)
            if is_surface_normal:
                tgt = transforms.functional.hflip(tgt)
                tgt[:, :, :1] = -tgt[:, :, :1]
            else:
                tgt = transforms.functional.hflip(tgt)

        return img, tgt


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        original_width, original_height = img.size
        pad_height = size - original_height if original_height < size else 0
        pad_width = size - original_width if original_width < size else 0
        img = functional.pad(img, (0, 0, pad_width, pad_height), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, cond_img=None):
        for t in self.transforms:
            image, target, cond_img = t(image, target, cond_img=cond_img)

        return image, target, cond_img


class Identity:
    def __init__(self):
        pass

    def __call__(self, image, target, cond_img=None):
        return image, target, cond_img


class PairedResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, cond_img):
        image = functional.resize(image, self.size)
        target = functional.resize(target, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        if cond_img is not None:
            cond_img = functional.resize(cond_img, self.size)

        return image, target, cond_img


class PairedRandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, cond_img=None):
        size = random.randint(self.min_size, self.max_size)
        image = functional.resize(image, size)
        target = functional.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
        if cond_img is not None:
            cond_img = functional.resize(cond_img, size)

        return image, target, cond_img


class PairedRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, cond_img=None):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=-1)
        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        # print('line 221', crop_params)
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)

        if cond_img is not None:
            cond_img = functional.crop(cond_img, *crop_params)

        return image, target, cond_img, crop_params


class PairedPILToTensor:
    def __call__(self, image, target, cond_img=None):
        image = functional.pil_to_tensor(image)
        target = np.array(target)
        if len(np.array(target).shape) == 2:
            target = torch.as_tensor(target, dtype=torch.int64)
        else:
            target = torch.as_tensor(target, dtype=torch.int64).permute(2,0,1)
        if cond_img is not None:
            cond_img = functional.pil_to_tensor(cond_img)

        return image, target, cond_img


class PILToTensor:
    def __call__(self, target):
        target = torch.as_tensor(np.array(target), dtype=torch.int64).permute(2,0,1)
        return target


class PairedConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target, cond_img=None):
        image = functional.convert_image_dtype(image, self.dtype)
        if cond_img is not None:
            cond_img = functional.convert_image_dtype(cond_img, self.dtype)

        return image, target, cond_img


class PairedNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, cond_img=None):
        image = functional.normalize(image, mean=self.mean, std=self.std)

        if cond_img is not None:
            cond_img = functional.normalize(cond_img, mean=self.mean, std=self.std)

        return image, target, cond_img


class PairedReduceLabels:
    def __call__(self, image, target, cond_img=None):
        if not isinstance(target, np.ndarray):
            target = np.array(target).astype(np.uint8)
        # avoid using underflow conversion
        target[target == 0] = 255
        target = target - 1
        target[target == 254] = 255
        target = Image.fromarray(target)


        return image, target, cond_img
