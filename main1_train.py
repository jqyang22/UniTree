#!/usr/bin/env python3

import os
import torch
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
available_gpus = list(range(torch.cuda.device_count()))

if torch.cuda.is_available():
    torch.zeros(1).cuda()

from core2.training_segcount_UniTree import trainer
from ptflops import get_model_complexity_info

from config import UniTreeTraining as configs



time0 = time.time()

config = configs.Configuration()
trainer_segcount = trainer(config, device=available_gpus)
trainer_segcount.vis()
trainer_segcount.train_config()
trainer_segcount.LOAD_model()

model = trainer_segcount.model
model.eval()

with torch.cuda.device(available_gpus[0]):
    macs, params = get_model_complexity_info(
        model,
        input_res=(4, 288, 288),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
print(f"FLOPs: {macs}")
print(f"Params: {params}", '\n')

trainer_segcount.train()

time1 = time.time()
print("Total time: ", time1 - time0, '(s)')
