from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


class Augmentation(object):
    """docstring for Augmentation"""
    def __init__(self):
        super(Augmentation, self).__init__()
        self.augmentation = self.strong_aug(p=0.6)


    def batch_augmentation(self, img, mask):
        data = {"image": img, "mask": mask}
        augmented = self.augmentation(**data)
        return augmented["image"], augmented["mask"]


    def strong_aug(self, p=0.5):
        return Compose([
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=10, p=0.7),
        ], p=p)
