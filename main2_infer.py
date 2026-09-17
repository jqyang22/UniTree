#!/usr/bin/env python3

import os
import ipdb
import torch
import numpy as np
from core2.inference_UniTree import anaer
from config import RasterAnalysis as RasterAnalysis
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
available_gpus = list(range(torch.cuda.device_count()))

print(torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

if torch.cuda.is_available():
    torch.zeros(1).cuda()

config = RasterAnalysis.Configuration()

predictor = anaer(config, device=available_gpus)
predictor.load_model()
predictor.segcount_RUN()
