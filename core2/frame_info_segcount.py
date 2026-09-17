#!/usr/bin/env python3

import numpy as np

def image_normalize(im, axis=(0, 1), c=1e-8):
    return (im - im.mean(axis)) / (im.std(axis) + c)

class FrameInfo:
    def __init__(self, img1, annotations, weight, density, dtype=np.float32):
        self.img1 = img1
        self.annotations = annotations
        self.weight = weight
        self.density = density
        self.dtype = dtype

    def getPatch(self, i, j, patch_size, img_size, normalize=1.0):
        patch1 = np.zeros(patch_size, dtype=self.dtype)
        im1 = self.img1[i:i+img_size[0], j:j+img_size[1]]
        r = np.random.random(1)
        if normalize >= r[0]:
            im1 = image_normalize(im1, axis=(0, 1))
        an = self.annotations[i:i+img_size[0], j:j+img_size[1]]
        an = np.expand_dims(an, axis=-1)
        we = self.weight[i:i+img_size[0], j:j+img_size[1]]
        we = np.expand_dims(we, axis=-1)
        den = self.density[i:i+img_size[0], j:j+img_size[1]]
        den = np.expand_dims(den, axis=-1)
        comb_img = np.concatenate((im1, an, we, den), axis=-1)
        patch1[:img_size[0], :img_size[1]] = comb_img
        return patch1

    def random_patch(self, patch_size, normalize):
        img_shape = self.img1.shape
        x = 0 if (img_shape[0] <= patch_size[0]) else np.random.randint(0, img_shape[0] - patch_size[0])
        y = 0 if (img_shape[1] <= patch_size[1]) else np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch1 = self.getPatch(x, y, patch_size, ic, normalize)
        return img_patch1



class FrameInfo_trg:
    def __init__(self, img1, dtype=np.float32):
        self.img1 = img1
        self.dtype = dtype

    def getPatch(self, i, j, patch_size, img_size, normalize=1.0):
        patch1 = np.zeros(patch_size, dtype=self.dtype)
        im1 = self.img1[i:i+img_size[0], j:j+img_size[1]]
        r = np.random.random(1)
        if normalize >= r[0]:
            im1 = image_normalize(im1, axis=(0, 1))

        comb_img = im1
        patch1[:img_size[0], :img_size[1]] = comb_img
        return patch1

    def random_patch(self, patch_size, normalize):
        img_shape = self.img1.shape
        x = 0 if (img_shape[0] <= patch_size[0]) else np.random.randint(0, img_shape[0] - patch_size[0])
        y = 0 if (img_shape[1] <= patch_size[1]) else np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch1 = self.getPatch(x, y, patch_size, ic, normalize)
        return img_patch1
