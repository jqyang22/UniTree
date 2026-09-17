import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Configuration:
    
    def __init__(self):

        self.input_image_dir = os.path.join(_REPO, 'data/test_input/')
        self.input_image_type = '.tif'
        self.input_image_pref = '' # prefix of image file names, can be used to filter out images
        self.channel_names1 = ['red', 'green', 'blue', 'infrared'] # if four color bands, set to ['red', 'green', 'blue', 'infrared']

        self.channels = [0, 1, 2, 3] # to take color bands in the correct order (match with the model)

        self.rgb2gray = 0 # set to 1 if using only grayscale image (convert rgb band to grayscale)
        self.band_switch = 0 # set to 1 if using only subset of bands or change the order of bands
        self.addndvi = 0
        self.trained_model_path = "./trained.pt"
        self.fillmiss = 0
        self.segcountpred = 1
        self.chmpred = 0
        self.normalize = 1
        self.segcount_tilenorm = 0
        self.maxnorm = 0
        self.gbnorm = 1
        self.gbnorm_FI = 0
        self.robustscale = 0
        self.robustscaleFI_local = 0
        self.robustscaleFI_gb = 0
        self.localtifnorm = 0
        self.multires = 0
        self.downsave = 0
        self.upsample = 0
        self.upscale = 0
        self.rescale_values = 0
        self.saveresult = 1
        self.tasks = 2
        self.change_input_size = 0
        self.input_size = 256
        self.input_shape = (self.input_size, self.input_size, len(self.channels))
        self.input_label_channel = [self.channels]
        self.inputBN = True

        self.output_dir = os.path.join(_REPO, 'data/test_pred/')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_suffix_seg = '_seg'
        self.output_suffix_density = '_density'
        self.chmdiff_prefix = 'diff_CHM_'
        self.output_image_type = '.tif'
        self.output_prefix = ''
        self.output_suffix_chm = '_chm'
        self.output_shapefile_type = '.shp'
        self.overwrite_analysed_files = False
        self.output_dtype='uint8'
        self.output_dtype_chm='int16'
        self.single_raster = 0
        self.aux_data = False
        self.operator = "MIX"
        self.threshold = 0.3 # for segmentation
        self.BATCH_SIZE = 1 # Depends upon GPU memory and WIDTH and HEIGHT
        self.WIDTH = 256
        self.HEIGHT = 256
        self.STRIDE = 246
        self.train_resolution = 0.2
        self.test_resolution = 0.6
