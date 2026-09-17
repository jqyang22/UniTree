import numpy as np

def image_normalize(im, axis = (0,1), c = 1e-8):
    return (im - im.mean(axis)) / (im.std(axis) + c)
   
 
class FrameInfo:

    def __init__(self, img, annotations, weight, dtype=np.float32):
        self.img = img
        self.annotations = annotations
        self.weight = weight
        self.dtype = dtype

    def getPatch(self, i, j, patch_size, img_size, normalize=1.0):
        patch = np.zeros(patch_size, dtype=self.dtype)
    
        im = self.img[i:i + img_size[0], j:j + img_size[1]]
        r = np.random.random(1)
        if normalize >= r[0]:
            im = image_normalize(im, axis=(0, 1))
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        an = np.expand_dims(an, axis=-1)
        we = self.weight[i:i + img_size[0], j:j + img_size[1]]
        we = np.expand_dims(we, axis=-1)
        comb_img = np.concatenate((im, an, we), axis=-1)
        patch[:img_size[0], :img_size[1], ] = comb_img
        return (patch)

    def sequential_patches(self, patch_size, step_size, normalize):
        img_shape = self.img.shape
        x = range(0, img_shape[0] - patch_size[0], step_size[0])
        y = range(0, img_shape[1] - patch_size[1], step_size[1])
        if (img_shape[0] <= patch_size[0]):
            x = [0]
        if (img_shape[1] <= patch_size[1]):
            y = [0]

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        for i, j in xy:
            img_patch = self.getPatch(i, j, patch_size, ic, normalize)
            img_patches.append(img_patch)
        return (img_patches)

    def random_patch(self, patch_size, normalize):
        img_shape = self.img.shape
        if (img_shape[0] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        if (img_shape[1] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch = self.getPatch(x, y, patch_size, ic, normalize)
        return (img_patch)