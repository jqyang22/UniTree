#!/usr/bin/env python3

import os
import rasterio
from rasterio import windows
from rasterio.enums import Resampling
import geopandas as gps
import numpy as np
from tqdm import tqdm
from itertools import product
import cv2
import math
import torch
import ipdb
import logging
from skimage.transform import resize

from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.frame_info import image_normalize

import matplotlib.pyplot as plt

class anaer:
    def __init__(self, config, device=None):
        self.config = config
        self.all_files = load_files(self.config)
        if device is not None and len(device) > 0:
            self.device = torch.device(f"cuda:{device[0]}")
            self.device_ids = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_ids = [0]

    def load_model(self):
        OPTIMIZER = adam
        if not self.config.change_input_size:
            if self.config.multires:
                from core2.UNet_multires_attention_segcount import UNet as UniTree
            else:
                from core2.UniTree_net import UniTree

            if len(self.config.input_shape) == 4:
                model_input_shape = (self.config.input_shape[3], self.config.input_shape[1], self.config.input_shape[2])
            elif len(self.config.input_shape) == 3:
                model_input_shape = (self.config.input_shape[2], self.config.input_shape[0], self.config.input_shape[1])
            else:
                raise ValueError("Unexpected input_shape dimension: {}".format(self.config.input_shape))

            self.model = UniTree(model_input_shape).to(self.device)


            state_dict = torch.load(self.config.trained_model_path, map_location=self.device, weights_only=True)
            if any(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '')
                    new_state_dict[new_key] = v
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict, strict=False)


            self.model.eval()
            self.model_chm = None
        else:
            raise NotImplementedError('change_input_size not supported in this conversion')
        logging.info('Model(s) loaded')

    def segcount_RUN(self):
        predict_ready_run(self.config, self.all_files, self.model, self.model_chm, self.config.output_dir, eva=0, th=self.config.threshold, rgb2gray=self.config.rgb2gray)
        return

def load_files(config):
    exclude = set(['water_new', 'md5', 'pred', 'test_kay'])
    all_files = []
    for root, dirs, files in os.walk(config.input_image_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            if file.endswith(config.input_image_type) and file.startswith(config.input_image_pref):
                all_files.append((os.path.join(root, file), file))
    print('Number of raw tif to predict:', len(all_files))
    if config.fillmiss:
        doneff = gps.read_file(config.grids)
        donef2 = list(doneff['filepath'])
        done_names= set([os.path.basename(f)[:6] for f in donef2])
        all_files = [f for f in all_files if os.path.splitext(f[1])[0] not in done_names]
    return all_files

def addTOResult(res, prediction, row, col, he, wi, operator='MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]

    mask = np.zeros_like(newPredictions)
    edge = 10
    mask[:edge, :] = 1
    mask[-edge:, :] = 1
    mask[:, :edge] = 1
    mask[:, -edge:] = 1

    fused = currValue.copy()

    if operator == 'MIN':
        fused[mask == 1] = np.minimum(currValue[mask == 1], newPredictions[mask == 1])
        fused[mask == 0] = newPredictions[mask == 0]

    elif operator == 'MAX':
        fused = np.maximum(currValue, newPredictions)
    elif operator == 'MEAN':
        fused[mask == 1] = (currValue[mask == 1] + newPredictions[mask == 1]) / 2.0
        fused[mask == 0] = newPredictions[mask == 0]

    else:
        fused = newPredictions

    res[row:row+he, col:col+wi] = fused
    return res


def visualize_density_map(density_map, method='log', cmap='hot', save_path=None):
    if method == 'log':
        vis = np.log1p(density_map)
    elif method == 'scale':
        vis = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-6)
        vis *= 3.0
        vis = np.clip(vis, 0, 1)
    else:
        vis = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-6)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis, cmap=cmap)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def addTOResult_chm(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
    if operator == 'MIN':
        currValue [currValue == -1] = 1
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    elif operator == "MIX":
        mm1 = currValue!=0
        try:
            currValue[mm1] = currValue[mm1] * 0.5 + newPredictions[mm1] * 0.5
            mm2 = (currValue==0)
            currValue[mm2] = newPredictions[mm2]
            resultant = currValue
        except:
            resultant = newPredictions[:256, :256]

    else:
        resultant = newPredictions
    res[row:row+he, col:col+wi] =  resultant
    return (res)

def predict_using_model_segcount_fi(model, batch, batch_pos, maskseg, maskdens, operator, upsample=1, downsave=1, upscale=2, rescale_values=1):
    model.eval()
    with torch.no_grad():
        tm1 = torch.from_numpy(np.stack(batch, axis=0)).permute(0, 3, 1, 2).float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        _, seg, _, dens, edge = model(tm1)
        seg = seg.cpu().numpy()
        dens = dens.cpu().numpy()

    for i, pos in enumerate(batch_pos):
        col, row, wi, he = pos
        p = np.squeeze(seg[i], axis=0)
        c = np.squeeze(dens[i], axis=0)

        maskseg = addTOResult(maskseg, p, row, col, he, wi, operator)
        maskdens = addTOResult(maskdens, c, row, col, he, wi, operator)

    return maskseg, maskdens

def predict_using_model_chm_fi(model, batch, batch_pos, mask, operator, upsample=1, downsave=1, upscale=2, rescale_values=1):
    model.eval()
    with torch.no_grad():
        if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            tm1 = []
            tm2 = []
            for p in batch:
                tm1.append(p[0])
                tm2.append(p[1])
            tm1 = torch.from_numpy(np.stack(tm1, axis=0)).permute(0, 3, 1, 2).float().to(model.device if hasattr(model, 'device') else torch.device('cpu'))
            tm2 = torch.from_numpy(np.stack(tm2, axis=0)).permute(0, 3, 1, 2).float().to(model.device if hasattr(model, 'device') else torch.device('cpu'))
            prediction = model([tm1, tm2])
        else:
            tm1 = torch.from_numpy(np.stack(batch, axis=0)).permute(0, 3, 1, 2).float().to(model.device if hasattr(model, 'device') else torch.device('cpu'))
            prediction = model(tm1)
        prediction = prediction.cpu().numpy()
    for i, pos in enumerate(batch_pos):
        col, row, wi, he = pos
        p = np.squeeze(prediction[i], axis=-1)
        mask = addTOResult_chm(mask, p, row, col, he, wi, operator)
    return mask

def detect_tree_segcount_fi(config, model, img, width=256, height=256, stride=128, normalize=True, auxData=0, singleRaster=1, multires=1, upsample=1, downsave=1, upscale=2, rescale_values=1, rgb2gray=0):
    if 'chm' in config.channel_names1:
        raise NotImplementedError('CHM as input not supported yet')
    else:
        CHM = 0
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()
    if config.segcount_tilenorm:
        print('tile norm')
        temp_imm = img.read()
        temp_imm = np.transpose(temp_imm, axes=(1, 2, 0))
        means = np.mean(temp_imm, axis=(0, 1))
        stds = np.std(temp_imm, axis=(0, 1))
    if 'float' not in meta['dtype']:
        meta['dtype'] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

    if downsave or not upsample:
        masksegs = np.zeros((nrows, nols), dtype=np.float32)
        maskdenss = np.zeros((nrows, nols), dtype=np.float32)
    elif not downsave:
        masksegs = np.zeros((int(nrows*upscale), int(nols*upscale)), dtype=np.float32)
        maskdenss = np.zeros((int(nrows*upscale), int(nols*upscale)), dtype=np.float32)
        meta.update({'width': int(nols*upscale), 'height': int(nrows*upscale)})
    if rgb2gray:
        meta.update({'count': 1})

    batch = []
    batch_pos = []
    for col_off, row_off in tqdm(offsets):
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        if upsample:
            patch1 = np.zeros((int(height*upscale), int(width*upscale), len(config.channels)))
            if config.band_switch:
                patch1 = np.zeros((int(height*upscale), int(width*upscale), int(len(config.channels))))
            temp_im1 = img.read(
                out_shape=(img.count, int(window.height*upscale), int(window.width*upscale)),
                resampling=Resampling.bilinear, window=window)
        else:
            patch1 = np.zeros((height, width, len(config.channels)))
            if config.band_switch:
                patch1 = np.zeros((height, width, int(len(config.channels))))
            temp_im1 = img.read(window=window)

        temp_im1 = np.transpose(temp_im1, axes=(1, 2, 0))
        try:
            temp_im1 = temp_im1[:, :, config.channels]
        except:
            ipdb.set_trace()

        if rgb2gray:
            temp_im1 = rgb2gray_convert(temp_im1)[..., np.newaxis]

        if config.segcount_tilenorm:
            temp_im1 = (temp_im1 - means) / stds

        if normalize:
            temp_im1 = image_normalize(temp_im1, axis=(0,1))

        if upsample:
            patch1[:int(window.height*upscale), :int(window.width*upscale)] = temp_im1
        else:
            patch1[:int(window.height), :int(window.width)] = temp_im1

        batch.append(patch1)

        if downsave or not upsample:
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
        elif not downsave:
            batch_pos.append((int(window.col_off*upscale), int(window.row_off*upscale), int(window.width*upscale), int(window.height*upscale)))
        if len(batch) == config.BATCH_SIZE:

            masksegs, maskdenss = predict_using_model_segcount_fi(model, batch, batch_pos, masksegs, maskdenss, 'MAX', upsample=upsample, downsave=downsave, upscale=upscale, rescale_values=rescale_values)
            batch = []
            batch_pos = []
    if batch:
        masksegs, maskdenss = predict_using_model_segcount_fi(model, batch, batch_pos, masksegs, maskdenss, 'MEAN', upsample=upsample, downsave=downsave, upscale=upscale, rescale_values=rescale_values)
        batch = []
        batch_pos = []
    return masksegs, maskdenss, meta

def detect_tree_rawtif_fi(config, model, img, channels, width=256, height=256, stride=128, normalize=0, maxnorm=0, upsample=1, downsave=1, upscale=2, rescale_values=1):
    nols, nrows = img.meta['width'], img.meta['height']
    meta = img.meta.copy()
    if 'float' not in meta['dtype']:
        meta['dtype'] = np.float32
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    if downsave and upsample:
        mask = np.zeros((int(nrows), int(nols)), dtype=meta['dtype'])
    else:
        mask = np.zeros((int(nrows/2), int(nols/2)), dtype=meta['dtype'])
    batch = []
    batch_pos = []
    for col_off, row_off in tqdm(offsets):
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        if upsample:
            if config.addndvi:
                patch = np.zeros((height*2, width*2, len(channels)+1))
            else:
                patch = np.zeros((height*2, width*2, len(channels)))
            read_im = img.read(
                out_shape=(img.count, int(window.height*2), int(window.width*2)),
                resampling=Resampling.bilinear, window=window)
        else:
            if config.addndvi:
                patch = np.zeros((height, width, len(channels)+1))
            else:
                patch = np.zeros((height, width, len(channels)))
            read_im = img.read(window=window)
        read_im = np.transpose(read_im, axes=(1,2,0))
        temp_im = read_im[:, :, channels]
        if config.addndvi:
            NDVI = (temp_im[:, :, -1].astype(float) - temp_im[:, :, 0].astype(float)) / (temp_im[:, :, -1].astype(float) + temp_im[:, :, 0].astype(float))
            NDVI = NDVI[..., np.newaxis]
            temp_im = np.append(temp_im, NDVI, axis=-1)
        if len(channels) > 3:
            if config.gbnorm:
                import logging
                logging.info('all bands - gb norm')
                temp_im = temp_im / 255
                temp_im = (temp_im - np.array([[0.317, 0.350, 0.321, 0.560, 0]])) / np.array([[0.985, 0.895, 0.703, 1.107, 1]])
            elif config.robustscale:
                if normalize:
                    temp_im = image_normalize(temp_im, axis=(0,1))
        elif len(channels) == 3:
            if channels[0] == 1:
                if config.robustscale:
                    import logging
                    logging.info('3 bands - robust scale - DK - gb')
                    temp_im = (temp_im - np.array([[73.0, 72.0, 145.0]])) / np.array([[40.0, 24.0, 37.0]])
                if config.gbnorm:
                    temp_im = temp_im / 255
                    temp_im = (temp_im - np.array([[0.317, 0.350, 0.321]])) / np.array([[0.985, 0.895, 0.703]])
        if upsample:
            patch[:int(window.height*upscale), :int(window.width*upscale)] = temp_im
        else:
            patch[:int(window.height), :int(window.width)] = temp_im
        batch.append(patch)
        if upsample:
            batch_pos.append((int(window.col_off), int(window.row_off), int(window.width), int(window.height)))
        elif not upsample:
            batch_pos.append((int(window.col_off/2), int(window.row_off/2), int(window.width/2), int(window.height/2)))
        if len(batch) == config.BATCH_SIZE:
            mask = predict_using_model_chm_fi(model, batch, batch_pos, mask, config.operator, upsample=upsample, downsave=downsave, upscale=upscale, rescale_values=rescale_values)
            batch = []
            batch_pos = []
    if batch:
        mask = predict_using_model_chm_fi(model, batch, batch_pos, mask, config.operator, upsample=upsample, downsave=downsave, upscale=upscale, rescale_values=rescale_values)
        batch = []
        batch_pos = []
    if maxnorm:
        mask = mask * 97.19
    return mask, meta

def predict_ready_run(config, all_files, model_segcount, model_chm, output_dir, eva=0, th=0.5, rgb2gray=0):
    counter = 1
    counts = {}
    outputFiles = []
    for fullPath, filename in tqdm(all_files):
        outputFile = os.path.join(output_dir, filename[:-4] + config.output_suffix_seg + config.output_image_type)
        print(outputFile)
        outputFile2 = outputFile.replace(config.output_suffix_seg, config.output_suffix_chm)
        if not os.path.exists(outputFile) or not os.path.exists(outputFile2):
            outputFiles.append(outputFile)
            with rasterio.open(fullPath) as img:
                if config.segcountpred:
                    print('creating file', outputFile)
                    detectedMaskSeg, detectedMaskDens, detectedMeta = detect_tree_segcount_fi(
                        config, model_segcount, img, width=config.WIDTH, height=config.HEIGHT, stride=config.STRIDE, normalize=config.normalize,
                        auxData=config.aux_data, singleRaster=config.single_raster, multires=config.multires, upsample=config.upsample,
                        downsave=config.downsave, upscale=config.upscale, rescale_values=config.rescale_values, rgb2gray=rgb2gray)
                    writeMaskToDisk(detectedMaskSeg, detectedMeta, outputFile, image_type=config.output_image_type, output_shapefile_type=config.output_shapefile_type, write_as_type=config.output_dtype, th=th, create_countors=False)
                    outputFile_density = outputFile.replace(config.output_suffix_seg, config.output_suffix_density)
                    writeMaskToDisk(detectedMaskDens, detectedMeta, outputFile_density, image_type=config.output_image_type, output_shapefile_type=config.output_shapefile_type, write_as_type='float32', th=th, create_countors=False, convert=0)

                    counts[filename] = detectedMaskDens.sum()
                if config.chmpred:
                    print('creating file', outputFile2)
                    detectedMaskChm, detectedMetaChm = detect_tree_rawtif_fi(
                        config, model_chm, img, config.channels, width=config.WIDTH, height=config.HEIGHT, stride=config.STRIDE, normalize=config.normalize, maxnorm=config.maxnorm, upsample=config.upsample,
                        downsave=config.downsave, upscale=config.upscale, rescale_values=config.rescale_values)
                    writeMaskToDiskChm(detectedMaskChm, detectedMetaChm, outputFile2, image_type=config.output_image_type, write_as_type=config.output_dtype_chm, scale=0)
            counter += 1
        else:
            print('Skipping: File already analysed!', fullPath)
    return counter

def writeMaskToDisk(detected_mask, detected_meta, wp, image_type, output_shapefile_type, write_as_type='uint8', th=0.5, create_countors=False, convert=1, rescale=0):
    meta = detected_meta.copy()
    if convert:
        if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
            print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
            detected_mask[detected_mask < th] = 0
            detected_mask[detected_mask >= th] = 1
    if rescale:
        detected_mask = detected_mask * 10000
    detected_mask = detected_mask.astype(write_as_type)
    if detected_mask.ndim != 2:
        detected_mask = detected_mask[0]
    meta['dtype'] = write_as_type
    meta['count'] = 1
    if rescale:
        meta.update({'compress': 'lzw', 'driver': 'GTiff', 'nodata': 32767})
    else:
        meta.update({'compress': 'lzw', 'driver': 'GTiff', 'nodata': 255})
    with rasterio.open(wp, 'w', **meta) as outds:
        outds.write(detected_mask, 1)
    return

def writeMaskToDiskChm(detected_mask, detected_meta, wp, image_type, write_as_type='float32', scale=1):
    print('range height', detected_mask.min(), detected_mask.max())
    if scale:
        detected_mask = detected_mask * 100
    print('mask', detected_mask.shape)
    print('meta', detected_meta['height'])
    detected_mask = detected_mask.astype(write_as_type)
    detected_meta['dtype'] = write_as_type
    detected_meta['count'] = 1
    detected_meta.update({'compress': 'lzw', 'nodata': 9999})
    try:
        with rasterio.open(wp, 'w', **detected_meta) as outds:
            outds.write(detected_mask, 1)
    except:
        detected_meta.update({'compress': 'lzw', 'nodata': 9999, 'driver': 'GTiff'})
        with rasterio.open(wp, 'w', **detected_meta) as outds:
            outds.write(detected_mask, 1)
    return

def rgb2gray_convert(rgb):
    print('rgb2gray', rgb.shape)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
