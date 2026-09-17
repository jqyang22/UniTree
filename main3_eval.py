import argparse
import os

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from sklearn.metrics import confusion_matrix

REPO = os.path.dirname(os.path.abspath(__file__))
INSIDE_ONLY = False
TEST_DIR = os.path.join(REPO, "data", "test")
RECT_SUFFIX = {"YSPark": "_seg_rectangle.shp"}
DEFAULT_RECT = "_seg_rectangle.shp"


def rasterize_rectangle_mask(rectangle_shp_path, reference_raster_path):
    rect_gdf = gpd.read_file(rectangle_shp_path)
    rect_gdf = rect_gdf[rect_gdf.is_valid & rect_gdf.geometry.notnull()]
    with rasterio.open(reference_raster_path) as src:
        shape = (src.height, src.width)
        transform = src.transform
    return rasterio.features.rasterize(
        [(geom, 1) for geom in rect_gdf.geometry],
        out_shape=shape, transform=transform, fill=0, dtype="uint8")


def load_masks(mask_result_tif_path, label_shp_path, rectangle_mask):
    with rasterio.open(mask_result_tif_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs
        mask_binary = (mask > 0).astype(np.uint8)
    label_gdf = gpd.read_file(label_shp_path).to_crs(crs)
    label_gdf = label_gdf[label_gdf.is_valid & label_gdf.geometry.notnull()]
    label_mask = rasterio.features.rasterize(
        [(geom, 1) for geom in label_gdf.geometry],
        out_shape=mask.shape, transform=transform, fill=0, dtype="uint8")
    return mask_binary * rectangle_mask, label_mask * rectangle_mask


def compute_metrics(pred, label, inside=None):
    if inside is not None:
        pred = pred[inside == 1]
        label = label[inside == 1]
    cm = confusion_matrix(label.flatten(), pred.flatten(), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    bg_iou = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0
    opr = fp / (fp + tn) if (fp + tn) > 0 else 0
    upr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return {"IoU (Foreground)": iou, "IoU (Background)": bg_iou, "mIoU": (iou + bg_iou) / 2,
            "Precision": precision, "Recall": recall, "F1 Score": f1,
            "OPR": opr, "UPR": upr}


def tile_paths(tile):
    lbl = os.path.join(TEST_DIR, tile, "lbl")
    return (os.path.join(TEST_DIR, tile, "pred", tile + "_seg.tif"),
            os.path.join(lbl, tile + "_seg_polygon.shp"),
            os.path.join(lbl, tile + RECT_SUFFIX.get(tile, DEFAULT_RECT)))


def run(pred_path, label_path, rect_path, name):
    rect = rasterize_rectangle_mask(rect_path, pred_path)
    pred_mask, label_mask = load_masks(pred_path, label_path, rect)
    m = compute_metrics(pred_mask, label_mask, inside=rect if INSIDE_ONLY else None)
    print("\n%s" % name)
    print("  pred : %s" % pred_path)
    for k, v in m.items():
        print("  %-18s %.4f" % (k, v))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile")
    ap.add_argument("--pred")
    ap.add_argument("--label")
    ap.add_argument("--rect")
    ap.add_argument("--inside-only", action="store_true",
                    help="score only pixels inside the rectangle instead of the whole raster")
    a = ap.parse_args()
    INSIDE_ONLY = a.inside_only
    if a.pred:
        run(a.pred, a.label, a.rect, "custom")
    else:
        tiles = [a.tile] if a.tile else sorted(os.listdir(TEST_DIR))
        for t in tiles:
            run(*tile_paths(t), name=t)
