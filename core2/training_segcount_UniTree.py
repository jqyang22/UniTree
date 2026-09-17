#!/usr/bin/env python3

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import numpy as np
from PIL import Image
import rasterio
import imgaug as ia

import imageio
import os
import time
import rasterio.warp

from core2.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity, miou, weight_miou
from core2.optimizers import adaDelta, adagrad, adam, nadam
from core2.split_frames import split_dataset
from core2.visualize import display_images

from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import bce_loss, prob_2_entropy
from advent.utils.loss import entropy_loss
import random
import copy
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_downsample(tensor, scale_factor):
    h, w = tensor.shape[-2:]

    new_h = int(round(h * scale_factor))
    new_w = int(round(w * scale_factor))

    tensor_resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear',
                                   align_corners=False)
    return tensor_resized

class trainer:
    def __init__(self, config, device=None):
        self.config = config
        self.source_loader, self.target_loader, self.no_frames = load_generators(self.config)
        if device is not None and len(device) > 0:
            self.device = torch.device(f"cuda:{device[0]}")
            self.device_ids = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_ids = [0]

    def vis(self):

        patch_visualizer(self.config, self.source_loader)

    def train_config(self):

        self.OPTIMIZER = adam
        self.LOSS_SEG = tversky
        self.LOSS_DENS = nn.MSELoss()

        timestr = time.strftime("%Y%m%d-%H%M")

        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path)

        model_path = os.path.join(self.config.model_path,'trees_{}_{}_{}_{}_{}_frames_{}_{}_{}weight_{}{}_densityR_{}-{}-{}.pt'.format(
                        timestr, self.config.OPTIMIZER_NAME, self.config.chs, self.config.input_shape[0], self.no_frames, self.config.LOSS_NAME,
                        self.config.LOSS2, self.config.boundary_weights, self.config.model_name, self.config.sufix, self.config.task_ratio[0],
                        self.config.task_ratio[1],self.config.task_ratio[2]))

        return model_path


    def LOAD_model(self):

        if self.config.multires:
            from core2.UNet_multires_attention_segcount import UNet as UniTree
        elif not self.config.multires:
            if not self.config.ifBN:
                from core2.UNet_attention_segcount_noBN import UNet as UniTree
            elif self.config.ifBN:
                from core2.UniTree_net import UniTree

        if len(self.config.input_shape) == 4:
            model_input_shape = (self.config.input_shape[3], self.config.input_shape[1], self.config.input_shape[2])
        elif len(self.config.input_shape) == 3:
            model_input_shape = (self.config.input_shape[2], self.config.input_shape[0], self.config.input_shape[1])
        else:
            raise ValueError("Unexpected input_shape dimension: {}".format(self.config.input_shape))

        self.model = UniTree(in_channels = model_input_shape)

        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        return self.model

    def train(self):

        model_path = trainer.train_config(self)
        train_advent_uda(self.OPTIMIZER, self.LOSS_SEG, self.LOSS_DENS, self.config, self.model, self.source_loader, self.target_loader, self.device, self.config.task_ratio, model_path, None)

        return


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        true_pos = (inputs * targets).sum(dim=(1, 2, 3))
        false_neg = ((1 - inputs) * targets).sum(dim=(1, 2, 3))
        false_pos = (inputs * (1 - targets)).sum(dim=(1, 2, 3))
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return 1 - tversky.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
def visualize_prediction(input_tensor, seg, edge):
    import matplotlib.pyplot as plt

    img = input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    seg = seg[0, 0].detach().cpu().numpy()
    edge = edge[0, 0].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[1].imshow(seg, cmap='gray')
    axs[1].set_title("Segmentation Output")
    axs[2].imshow(edge, cmap='gray')
    axs[2].set_title("Edge Output")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visual_edge.png'))
    plt.show()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        if targets.size(1) == 2:
            targets = targets[:, 0:1, :, :]
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

def mask_to_edge(mask, kernel_size=3):
    edge_x = F.pad(mask[:, :, 1:, :] != mask[:, :, :-1, :], (0, 0, 1, 0)).float()
    edge_y = F.pad(mask[:, :, :, 1:] != mask[:, :, :, :-1], (1, 0, 0, 0)).float()
    edge = (edge_x + edge_y).clamp(0, 1)
    return edge


def train_advent_uda(OPTIMIZER, LOSS_SEG, LOSS_DENS, config, model, source_loader, target_loader, device,
                       task_ratio, model_path, callbacks_list):

    print(f"Model_path saved: {model_path}", '\n')

    phases = [
        {"start_epoch": 0, "end_epoch": 100, "density_weight": 100},
        {"start_epoch": 101, "end_epoch": 500, "density_weight": 10000},
        {"start_epoch": 501, "end_epoch": 1500, "density_weight": 100000},

    ]

    best_output_seg_weight_miou = 0.0
    best_output_seg_weight_miou_kd = 0.0
    best_output_seg_weight_miou_uda = 0.0
    model_path_final = model_path[:-3] + f'_final.pt'

    ema_decay = getattr(config, 'ema_decay', 0.99)

    teacher_model = copy.deepcopy(model)
    teacher_model.eval()

    d_main = get_fc_discriminator(num_classes=1).to(device)

    OPTIMIZER = OPTIMIZER(model.parameters())
    optimizer_d = optim.Adam(d_main.parameters(), lr=0.00001, betas=(0.9, 0.99))

    source_label = 0
    target_label = 1

    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

    for phase in phases:
        print(
            f"Starting phase: Epochs {phase['start_epoch']} to {phase['end_epoch']} with density weight {phase['density_weight']}", '\n')

        for epoch in range(phase["start_epoch"], phase["end_epoch"] + 1):
            running_loss = 0.0

            pbar = tqdm(range(config.MAX_TRAIN_STEPS), desc=f"Epoch {epoch}/{phase['end_epoch']}", ncols=200)

            uda_enabled = (epoch >= config.uda_warmup_epoch) and (epoch < config.uda_stop_epoch)
            kd_enabled = (epoch >= config.kd_warmup_epoch) and (epoch < config.kd_stop_epoch)

            model.train()
            d_main.train()

            for step, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):

                if step >= config.MAX_TRAIN_STEPS:
                    break

                src_inputs = source_data[0]
                src_labels = source_data[1]

                if isinstance(src_inputs, list):
                    src_inputs = [torch.tensor(x).permute(0, 3, 1, 2).float().to(device) for x in src_inputs]
                    if len(src_inputs) == 1:
                        src_inputs = src_inputs[0]
                else:
                    src_inputs = torch.tensor(src_inputs).permute(0, 3, 1, 2).float().to(device)

                src_lbl_seg = torch.tensor(src_labels['output_seg']).permute(0, 3, 1, 2).float().to(device)
                src_lbl_dense = torch.tensor(src_labels['output_dens']).unsqueeze(1).squeeze(-1).float().to(device)


                src_inputs_02m = src_inputs

                scale = 1 / 3
                src_inputs_06m = safe_downsample(src_inputs, scale)
                src_lbl_seg_06m = safe_downsample(src_lbl_seg, scale)
                src_lbl_dense_06m = safe_downsample(src_lbl_dense, scale)


                trg_inputs = target_data[0]
                if isinstance(trg_inputs, list):
                    trg_inputs = [torch.tensor(x).permute(0, 3, 1, 2).float().to(device) for x in trg_inputs]
                    if len(trg_inputs) == 1:
                        trg_inputs = trg_inputs[0]
                else:
                    trg_inputs = torch.tensor(trg_inputs).permute(0, 3, 1, 2).float().to(device)

                trg_inputs_06m = trg_inputs

                trg_inputs_flip = torch.flip(trg_inputs_06m, dims=[3])


                OPTIMIZER.zero_grad()
                optimizer_d.zero_grad()

                for param in d_main.parameters():
                    param.requires_grad = False


                if isinstance(src_inputs, list):
                    pred_src_seg, pred_src_dense = model((src_inputs[0], src_inputs[1]))
                else:

                    bottleneck_feature, pred_src_seg_06m, pred_src_logits_06m, pred_src_dense_06m, pred_src_edge_06m = model(src_inputs_06m)

                    with torch.no_grad():
                        bottleneck_feature_02m, pred_teacher_seg_02m, pred_teacher_logits_02m, pred_teacher_dense_02m, _ = teacher_model(src_inputs_02m)

                    pred_teacher_logits_resized = F.interpolate(pred_teacher_logits_02m, size=pred_src_logits_06m.shape[-2:],
                                                                mode='bilinear', align_corners=False)

                    pred_teacher_dense_resized = F.interpolate(pred_teacher_dense_02m, size=pred_src_dense_06m.shape[-2:],
                                                               mode='bilinear', align_corners=False)

                    if epoch < 200:
                        loss_seg = 0.5 * focal_loss(pred_src_logits_06m,
                                                    src_lbl_seg_06m[:, 0:1, :, :]) + 0.5 * LOSS_SEG(src_lbl_seg_06m, pred_src_seg_06m)
                    else:
                        loss_seg = LOSS_SEG(src_lbl_seg_06m, pred_src_seg_06m)

                    LOSS_EDGE = DiceLoss()(pred_src_edge_06m, mask_to_edge(src_lbl_seg_06m[:, 0:1, :, :]))


                    loss_dens = LOSS_DENS(src_lbl_dense_06m, pred_src_dense_06m)

                    if kd_enabled:
                        T = 3.0
                        loss_kd_seg = F.binary_cross_entropy_with_logits(
                            pred_src_logits_06m / T,
                            torch.sigmoid(pred_teacher_logits_resized / T).detach()
                        ) * (T * T)

                        scale_ratio = 0.6 / 0.2
                        pred_teacher_dense_resized = pred_teacher_dense_resized * (scale_ratio ** 2)
                        pred_teacher_dense_resized = torch.clamp(pred_teacher_dense_resized, 0, 10)
                        pred_src_dense_06m = torch.clamp(pred_src_dense_06m, 0, 10)
                        pred_teacher_dense_resized = pred_teacher_dense_resized / 10.0
                        pred_src_dense_06m = pred_src_dense_06m / 10.0
                        loss_kd_dens = F.smooth_l1_loss(pred_src_dense_06m, pred_teacher_dense_resized.detach())

                        loss_kd_feature = F.smooth_l1_loss(
                            bottleneck_feature,
                            F.interpolate(bottleneck_feature_02m, size=bottleneck_feature.shape[-2:], mode='bilinear',
                                          align_corners=False)
                        )

                    else:
                        loss_kd_seg = torch.tensor(0.0, device=device)
                        loss_kd_dens = torch.tensor(0.0, device=device)
                        loss_kd_feature = torch.tensor(0.0, device=device)

                    loss = loss_seg + phase["density_weight"] * loss_dens + 0.01 * LOSS_EDGE + config.lambda_kd * (loss_kd_seg + loss_kd_dens) + 0.0001 * loss_kd_feature

                    loss.backward()

                    if uda_enabled:
                        _, pred_trg_seg_ori, pred_trg_logits, _, _ = model(trg_inputs_06m)
                        pred_trg_prob = torch.sigmoid(pred_trg_logits)
                        entropy_trg = - (pred_trg_prob * torch.log(pred_trg_prob + 1e-6) + (1 - pred_trg_prob) * torch.log(
                            1 - pred_trg_prob + 1e-6))
                        d_out_trg = d_main(entropy_trg)
                        loss_adv = bce_loss(d_out_trg, source_label)

                        prob_np = pred_trg_prob[0, 0].cpu().detach().numpy()
                        entropy_np = entropy_trg[0, 0].cpu().detach().numpy()

                        input_np = trg_inputs_06m[0].cpu().detach().numpy()
                        input_np = np.transpose(input_np, (1, 2, 0))

                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])

                        input_np = input_np[:, :, :3]

                        input_np = input_np * std + mean
                        input_np = np.clip(input_np, 0, 1)

                        plt.figure(figsize=(15, 4))

                        plt.subplot(1, 3, 1)
                        plt.title("Input Image")
                        plt.imshow(input_np)
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        plt.title("Predicted Probability")
                        plt.imshow(prob_np, cmap='viridis')
                        plt.axis('off')
                        plt.colorbar()

                        plt.subplot(1, 3, 3)
                        plt.title("Entropy (Uncertainty)")
                        plt.imshow(entropy_np, cmap='hot')
                        plt.axis('off')
                        plt.colorbar()

                        plt.tight_layout()
                        plt.savefig(os.path.dirname(model_path) + '/mid_feature/adv_mid_map_epoch' + str(epoch) + '.png')
                        plt.show()

                        pred_trg_seg_aug = model(trg_inputs_flip)[1]
                        pred_trg_aug_flip_back = torch.flip(pred_trg_seg_aug, dims=[3])

                        (config.LAMBDA_ADV * loss_adv).backward()

                    OPTIMIZER.step()



                    with torch.no_grad():
                        for t_param, s_param in zip(teacher_model.parameters(), model.parameters()):
                            t_param.data = ema_decay * t_param.data + (1 - ema_decay) * s_param.data

                    if uda_enabled:
                        for param in d_main.parameters():
                            param.requires_grad = True

                        pred_src_prob = torch.sigmoid(pred_src_logits_06m.detach())
                        entropy_src = - (pred_src_prob * torch.log(pred_src_prob + 1e-6) + (1 - pred_src_prob) * torch.log(
                            1 - pred_src_prob + 1e-6))
                        d_out_src = d_main(entropy_src)

                        loss_d_src = bce_loss(d_out_src, source_label)
                        (config.LAMBDA_ADV * loss_d_src * 0.5).backward()

                        pred_trg_prob = torch.sigmoid(pred_trg_logits.detach())
                        entropy_trg = - (pred_trg_prob * torch.log(pred_trg_prob + 1e-6) + (1 - pred_trg_prob) * torch.log(
                            1 - pred_trg_prob + 1e-6))
                        d_out_trg = d_main(entropy_trg)

                        loss_d_trg = bce_loss(d_out_trg, target_label)
                        (config.LAMBDA_ADV * loss_d_trg * 0.5).backward()

                        optimizer_d.step()


                running_loss += loss.item() * src_lbl_seg_06m.size(0)

                if step == (config.MAX_TRAIN_STEPS - 1):
                    dice = dice_coef(src_lbl_seg_06m, pred_src_seg_06m).item()
                    spec = specificity(src_lbl_seg_06m, pred_src_seg_06m)
                    sens = sensitivity(src_lbl_seg_06m, pred_src_seg_06m)
                    fscore = 2 * spec * sens / (spec + sens)
                    iou_val = miou(src_lbl_seg_06m, pred_src_seg_06m, device)
                    w_iou_val = weight_miou(src_lbl_seg_06m, pred_src_seg_06m)
                    acc = accuracy(src_lbl_seg_06m, pred_src_seg_06m)
                    rmse = torch.sqrt(loss_dens)

                    print(f"\nEpoch {epoch}/{phase['end_epoch']}, Step {step}/{config.MAX_TRAIN_STEPS}, "
                          f"Total Loss: {loss.item():.4f}\n "

                          f"{'AdvLoss: {:.4f}, DLoss_src: {:.4f}, DLoss_trg: {:.4f}, '.format(loss_adv.item(), loss_d_src.item(), loss_d_trg.item()) if uda_enabled else 'WARMUP (no D)'}\n",

                          f"output_seg_loss: {loss_seg.item():.4f}, "
                          f"output_dens_loss: {loss_dens.item():.4e}, "
                          f"output_seg_dice_coef: {dice:.4f}, "
                          f"output_seg_specificity: {spec:.4f}, "
                          f"output_seg_sensitivity: {sens:.4f}, "
                          f"output_seg_miou: {iou_val:.4f}, "
                          f"output_seg_weight_miou: {w_iou_val:.4f}, "
                          f"output_seg_accuracy: {acc:.4f}, "
                          f"output_dens_root_mean_squared_error: {rmse.item():.4f},\n "

                          f"{'KD_Loss_seg: {:.4f}, KD_Loss_dens: {:.4e}, loss_kd_feature: {:.4f}'.format(loss_kd_seg.item(), loss_kd_dens.item(), loss_kd_feature.item()) if kd_enabled else 'KD WARMUP (no KDD)'}\n",

                          )

                    torch.cuda.empty_cache()

                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "seg_loss": f"{loss_seg.item():.4f}",
                    "dens_loss": f"{phase['density_weight'] * loss_dens.item():.4e}\n",
                    "edge_loss": f"{0.01 * LOSS_EDGE.item():.4f}\n",
                    "loss_kd_seg": f"{config.lambda_kd * loss_kd_seg.item():.4f}\n",
                    "loss_kd_dens": f"{config.lambda_kd * loss_kd_dens.item():.4e}\n",
                    "loss_kd_feature": f"{0.0001 * loss_kd_feature.item():.4e}\n",
                })

                pbar.update(1)

            pbar.close()

            epoch_loss = running_loss / (config.MAX_TRAIN_STEPS * config.BATCH_SIZE)

            torch.cuda.empty_cache()

            torch.save(model.state_dict(), model_path_final)

            if w_iou_val > best_output_seg_weight_miou:
                best_output_seg_weight_miou = w_iou_val
                torch.save(model.state_dict(), model_path)
            if uda_enabled:
                if w_iou_val > best_output_seg_weight_miou_uda:
                    best_output_seg_weight_miou_uda = w_iou_val
                    model_path_uda = model_path[:-3] + f'_UDA.pt'
                    torch.save(model.state_dict(), model_path_uda)
            if kd_enabled:
                if w_iou_val > best_output_seg_weight_miou_kd:
                    best_output_seg_weight_miou_kd = w_iou_val
                    model_path_kd = model_path[:-3] + f'_KD.pt'
                    torch.save(model.state_dict(), model_path_kd)

    return


def validate(model, val_generator, density_weight, device, LOSS_SEG, LOSS_DENS, config):
    model.eval()
    running_loss = 0.0
    metrics_sum = {"dice": 0.0, "spec": 0.0, "sens": 0.0, "iou": 0.0, "w_iou": 0.0, "acc": 0.0, "rmse": 0.0}

    pbar = tqdm(range(config.VALID_IMG_COUNT), desc="Validation", ncols=100)

    with torch.no_grad():
        for step, batch in enumerate(val_generator):
            if step >= config.VALID_IMG_COUNT:
                break  

            inputs = batch[0]
            labels = batch[1]

            if isinstance(inputs, list):
                inputs = [torch.tensor(x).permute(0, 3, 1, 2).float().to(device) for x in inputs]
                if len(inputs) == 1:
                    inputs = inputs[0]
            else:
                inputs = torch.tensor(inputs).permute(0, 3, 1, 2).float().to(device)

            output_seg_true = torch.tensor(labels['output_seg']).permute(0, 3, 1, 2).float().to(device)
            output_dens_true = torch.tensor(labels['output_dens']).unsqueeze(1).float().to(device)
            output_dens_true = output_dens_true.squeeze(-1)

            if isinstance(inputs, list):
                output_seg_pred, output_dens_pred = model((inputs[0], inputs[1]))
            else:

                scale = 1 / 3
                inputs = safe_downsample(inputs, scale)
                output_dens_true = safe_downsample(output_dens_true, scale)
                output_seg_true = safe_downsample(output_seg_true, scale)

                output_seg_pred, _, output_dens_pred, output_edge_pred = model(inputs)

            LOSS_EDGE = DiceLoss()(output_edge_pred, mask_to_edge(output_seg_true[:, 0:1, :, :]))

            loss_seg = LOSS_SEG(output_seg_true, output_seg_pred)
            loss_dens = LOSS_DENS(output_dens_true, output_dens_pred)
            loss = loss_seg + density_weight * loss_dens

            loss = loss_seg + density_weight * loss_dens + LOSS_EDGE

            running_loss += loss.item()

            dice = dice_coef(output_seg_true, output_seg_pred).item()
            spec = specificity(output_seg_true, output_seg_pred)
            sens = sensitivity(output_seg_true, output_seg_pred)
            iou_val = miou(output_seg_true, output_seg_pred, device)
            w_iou_val = weight_miou(output_seg_true, output_seg_pred)
            acc = accuracy(output_seg_true, output_seg_pred)
            rmse = torch.sqrt(loss_dens)

            metrics_sum["dice"] += dice
            metrics_sum["spec"] += spec
            metrics_sum["sens"] += sens
            metrics_sum["iou"] += iou_val
            metrics_sum["w_iou"] += w_iou_val
            metrics_sum["acc"] += acc
            metrics_sum["rmse"] += rmse.item()

            torch.cuda.empty_cache()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "seg_loss": f"{loss_seg.item():.4f}",
                "dens_loss": f"{loss_dens.item():.4e}",
                "Dice": f"{dice:.4f}",
                "IoU": f"{iou_val:.4f}",
                "Acc": f"{acc:.4f}"
            })
            pbar.update(1)

    pbar.close()

    val_loss = running_loss / config.VALID_IMG_COUNT
    avg_metrics = {key: val / config.VALID_IMG_COUNT for key, val in metrics_sum.items()}

    print(f"Validation Results - Loss: {val_loss:.4f}, "
          f"Dice: {avg_metrics['dice']:.4f}, "
          f"Specificity: {avg_metrics['spec']:.4f}, "
          f"Sensitivity: {avg_metrics['sens']:.4f}, "
          f"IoU: {avg_metrics['iou']:.4f}, "
          f"Weighted IoU: {avg_metrics['w_iou']:.4f}, "
          f"Accuracy: {avg_metrics['acc']:.4f}, "
          f"RMSE: {avg_metrics['rmse']:.4f}")

    return val_loss, avg_metrics


def load_generators(config):
    if config.multires:
        from core2.frame_info_multires_segcount import FrameInfo
        from core2.dataset_generator_multires_segcount import DataGenerator
    elif not config.multires:
        from core2.frame_info_segcount import FrameInfo, FrameInfo_trg
        from core2.dataset_generator_segcount import DataGenerator

    frames_src = []

    all_files = os.listdir(config.base_dir)
    all_files_c1 = [fn for fn in all_files if fn.startswith(config.channel_names[0]) and fn.endswith(config.image_type)]

    for i, fn in enumerate(all_files_c1):
        img1 = rasterio.open(os.path.join(config.base_dir, fn)).read()
        if config.single_raster or not config.aux_data:
            for c in range(len(config.channel_names)-1):
                img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.channel_names[c+1]))).read(), axis = 0)

        else:
            print('Multi raster with aux data')
            for c in range(len(config.channel_names)-1):

                img1 = np.append(img1, rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.channel_names[c+1]))).read(), axis = 0)
            if config.multires:
                img2 = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.channel_names2[0]))).read()


        img1 = np.transpose(img1, axes=(1,2,0))
        if config.multires:

            img2 = np.transpose(img2, axes=(1,2,0))

        if config.grayscale:
            print('Using grayscale images!')
            img1 = rgb2gray(img1)
            img1 = img1[..., np.newaxis]
        annotation = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.annotation_fn))).read()
        annotation = np.squeeze(annotation)
        weight = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.weight_fn))).read()
        weight = np.squeeze(weight)
        density = rasterio.open(os.path.join(config.base_dir, fn.replace(config.channel_names[0],config.density_fn))).read()
        density = np.squeeze(density)
        if config.multires:
            f = FrameInfo(img1, img2, annotation, weight, density)
        elif not config.multires:
            f = FrameInfo(img1, annotation, weight, density)

        frames_src.append(f)


    frames_trg = []

    all_files = os.listdir(config.trg_dir)
    all_files_c1 = [fn for fn in all_files if fn.startswith(config.channel_names[0]) and fn.endswith(config.image_type)]
    print('trg: ', all_files_c1)

    for i, fn in enumerate(all_files_c1):
        img1 = rasterio.open(os.path.join(config.trg_dir, fn)).read()
        if config.single_raster or not config.aux_data:
            for c in range(len(config.channel_names)-1):
                img1 = np.append(img1, rasterio.open(os.path.join(config.trg_dir, fn.replace(config.channel_names[0],config.channel_names[c+1]))).read(), axis = 0)

        else:
            print('Multi raster with aux data')
            for c in range(len(config.channel_names)-1):

                img1 = np.append(img1, rasterio.open(os.path.join(config.trg_dir, fn.replace(config.channel_names[0],config.channel_names[c+1]))).read(), axis = 0)
            if config.multires:
                img2 = rasterio.open(os.path.join(config.trg_dir, fn.replace(config.channel_names[0],config.channel_names2[0]))).read()


        img1 = np.transpose(img1, axes=(1,2,0))
        if config.multires:

            img2 = np.transpose(img2, axes=(1,2,0))

        if config.grayscale:
            print('Using grayscale images!!!!')
            img1 = rgb2gray(img1)
            img1 = img1[..., np.newaxis]
        annotation = rasterio.open(os.path.join(config.trg_dir, fn.replace(config.channel_names[0],config.annotation_fn))).read()
        annotation = np.squeeze(annotation)
        weight = rasterio.open(os.path.join(config.trg_dir, fn.replace(config.channel_names[0],config.weight_fn))).read()
        weight = np.squeeze(weight)
        density = rasterio.open(os.path.join(config.trg_dir, fn.replace(config.channel_names[0],config.density_fn))).read()
        density = np.squeeze(density)
        if config.multires:
            f = FrameInfo(img1, img2, annotation, weight, density)
        elif not config.multires:
            f = FrameInfo(img1, annotation, weight, density)

        frames_trg.append(f)


    training_frames = list(range(len(frames_src)))

    validation_frames  = list(range(len(frames_trg)))
    print('Number of training frames:', len(frames_src), len(frames_trg))

    annotation_channels = config.input_label_channel + config.input_weight_channel + config.input_density_channel
    train_generator = DataGenerator(config.input_image_channel, config.patch_size, training_frames, frames_src, annotation_channels, config.boundary_weights, augmenter = 'iaa').random_generator(config.BATCH_SIZE, normalize = config.normalize)

    val_generator = DataGenerator(config.input_image_channel, config.patch_size, validation_frames, frames_trg, annotation_channels, config.boundary_weights, augmenter= None).random_generator(config.BATCH_SIZE, normalize = config.normalize)

    return train_generator, val_generator, len(all_files_c1)


def patch_visualizer(config, train_generator):
    for _ in range(1):
        train_images, real_label = next(train_generator)

        if config.multires:
            train_im1, train_im2 = train_images
            chms = train_im2[..., -1]
            print('CHM range:', chms.min(), chms.max())
        else:
            train_im1 = train_images

        if isinstance(train_im1, torch.Tensor):
            train_im1 = train_im1.cpu().numpy()

        print('Time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('Color Mean:', train_im1.mean(axis=(0, 1, 2)))
        print('Color Std:', train_im1.std(axis=(0, 1, 2)))
        print('Color Max:', train_im1.max(axis=(0, 1, 2)))

        if config.multires:
            if isinstance(train_im2, torch.Tensor):
                train_im2 = train_im2.cpu().numpy()
            print('Train_im2 Mean:', train_im2.mean(axis=(0, 1, 2)))
            print('Train_im2 Std:', train_im2.std(axis=(0, 1, 2)))
            train_im2 = resize(train_im2, (config.BATCH_SIZE, train_im1.shape[1], train_im1.shape[2]))

        output_dens = real_label['output_dens']
        output_seg = real_label['output_seg']

        if isinstance(output_dens, torch.Tensor):
            output_dens = output_dens.cpu().numpy()
        if isinstance(output_seg, torch.Tensor):
            output_seg = output_seg.cpu().numpy()

        print('Count:', output_dens.sum(axis=(1, 2)))
        print('Density map pixel value range:', output_dens.max() - output_dens.min())
        print('output_seg shape:', output_seg.shape)

        ann = output_seg[..., 0]
        wei = output_seg[..., 1]
        print('Boundary highlighted weights:', np.unique(wei))

        overlay = ann + wei
        overlay = overlay[..., np.newaxis]
        print('Seg mask unique values:', np.unique(ann))

        if config.multires:
            display_images(
                np.concatenate((train_im1, train_im2, output_seg, overlay, output_dens), axis=-1),
                savefig=True,
                savename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trn_img_multires')
            )
        else:
            display_images(
                np.concatenate((train_im1, output_seg, overlay, output_dens), axis=-1),
                savefig=True,
                savename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trn_img')
            )

    return


class LossWeightAdjust:
    def __init__(self, alpha=0.000001):
        self.alpha = alpha
        self.alphas = []
    def on_epoch_end(self, epoch, val_output_dens_loss):
        lam = 10 ** (-np.floor(np.log10(val_output_dens_loss)) - 2)
        self.alpha = lam
        logger.info("------- Loss weights recalibrated to alpha = %s -------" % self.alpha)
        print("------- Loss weights recalibrated to alpha = %s -------" % self.alpha)
        self.alphas.append(self.alpha)


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
