import os
from functools import reduce

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Configuration:
    def __init__(self):
        self.ntasks = 2
        self.multires = 0
        self.base_dir = os.path.join(_REPO, 'data/source/') # extracted image and label patches
        self.trg_dir = os.path.join(_REPO, 'data/target/') # extracted image and label patches
        self.image_type = '.png'
        self.grayscale = 0 # convert to grayscale
        self.channel_names = ['red', 'green', 'blue', 'infrared']
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary'
        self.density_fn = 'ann_kernel'
        self.boundary_weights = 5
        self.single_raster = False
        self.aux_data = 0
        self.patch_generation_stratergy = 'random'
        if self.grayscale:
            self.image_channels = 1
        else:
            self.image_channels = len(self.channel_names)
        self.all_channels = self.image_channels + 3
        self.patch_size = (288,288,self.all_channels)
        self.normalize = 0 # set to 0 if using input batch normalization

        self.upscale_factor = 2
        self.input_shape = (288,288,self.image_channels)  # for downsampling
        self.input_image_channel = list(range(self.image_channels))
        self.input_label_channel = [self.image_channels]
        self.input_weight_channel = [self.image_channels+1]
        self.input_density_channel = [self.image_channels+2]

        self.inputBN = 1
        self.task_ratio = [100, 1000, 10000] # list of 3 indicating the loss weighting ratio

        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 1000 # number of epochs
        self.ifBN = True # use batch norm

        self.VALID_IMG_COUNT = 100
        self.MAX_TRAIN_STEPS = 500

        self.OPTIMIZER_NAME = 'Adam_e4'
        self.LOSS_NAME = 'WTversky'
        self.LOSS2 = 'Mse'
        self.model_name = 'UniTree'
        self.sufix = ''
        self.chs = reduce(lambda a,b: a+str(b), self.channel_names, '')

        self.model_path = os.path.join(_REPO, './path')

        self.DA_METHOD = "AdvEnt"
        self.LAMBDA_ADV = 0.0001
        self.LAMBDA_ENT = 0.01
        self.uda_warmup_epoch = 500
        self.uda_stop_epoch = 800
        self.lambda_consistency = 0.01
        self.kd_warmup_epoch = 700
        self.kd_stop_epoch = 900
        self.lambda_kd = 0.0001