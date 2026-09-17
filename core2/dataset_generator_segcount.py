import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import torch

def imageAugmentationWithIAA():
    sometimes = lambda aug, prob=0.5: iaa.Sometimes(prob, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        sometimes(iaa.Crop(percent=(0, 0.1))),
    ], random_order=True)
    return seq

class DataGenerator():

    def __init__(self, input_image_channel1, patch_size, frame_list, frames, annotation_channel, boundary_weights=10, augmenter=None):
        self.input_image_channel1 = input_image_channel1
        self.patch_size = patch_size
        self.frame_list = frame_list
        self.frames = frames
        self.annotation_channel = annotation_channel
        self.boundary_weights = boundary_weights
        self.augmenter = augmenter

    def random_patch(self, BATCH_SIZE, normalize):
        patches1 = []

        for i in range(BATCH_SIZE):
            fn = np.random.choice(self.frame_list)
            frame = self.frames[fn]
            patch1 = frame.random_patch(self.patch_size, normalize)
            patches1.append(patch1)

        data1 = np.array(patches1)

        img1 = data1[..., self.input_image_channel1]

        ann_joint = data1[..., self.annotation_channel]
        return (img1, ann_joint)


    def random_generator(self, BATCH_SIZE, normalize=1):
        seq = imageAugmentationWithIAA()
        while True:
            X, y = self.random_patch(BATCH_SIZE, normalize)
            if self.augmenter == 'iaa':
                seq_det = seq.to_deterministic()
                X1 = seq_det.augment_images(X)
                y = seq_det.augment_images(y)


                ann = y[..., [0]]
                ann[ann < 0.5] = 0
                ann[ann >= 0.5] = 1
                weights = y[..., [1]]
                weights[weights >= 0.5] = self.boundary_weights
                weights[weights < 0.5] = 1
                density = y[..., [2]]

                ann_joint = np.concatenate((ann, weights), axis=-1)

                X1 = torch.from_numpy(X1).float()
                ann_joint = torch.from_numpy(ann_joint).float()
                density = torch.from_numpy(density).float()

                yield (X1, {'output_seg': ann_joint, 'output_dens': density})
            else:
                ann = y[..., [0]]
                weights = y[..., [1]]
                weights[weights >= 0.5] = self.boundary_weights
                weights[weights < 0.5] = 1
                density = y[..., [2]]
                ann_joint = np.concatenate((ann, weights), axis=-1)
                X = torch.from_numpy(X).float()
                ann_joint = torch.from_numpy(ann_joint).float()
                density = torch.from_numpy(density).float()
                yield (X, {'output_seg': ann_joint, 'output_dens': density})
