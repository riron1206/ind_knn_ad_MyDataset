"""
Gradioで推論
Usage:
    gradio ./gradio_app.py
    http://0.0.0.0:7860 からアクセスできる
    ※ctrl+cで終了するとプロセス残り続けgpuが開放されない。必ずdocker起動し直すこと
"""
import gradio as gr
import argparse
import os
import sys
import gc
import glob
import random
import pickle
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

sys.path.append('./ind_knn_ad/indad')
from models import SPADE, PaDiM, PatchCore

from run_mydataset import seed_everything, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()  # pytorch >= 1.9
def inference(image: torch.Tensor, device: str = device):
    """モデル推論"""
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        # ind_knn_ad
        # https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py#L65
        # score, segmentation_map = 異常度と特徴量マップ
        img_lvl_anom_score, pxl_lvl_anom_score = model.predict(image)
    img_lvl_anom_score = img_lvl_anom_score.cpu().detach().numpy()
    pxl_lvl_anom_score = pxl_lvl_anom_score.cpu().detach().numpy()
    return img_lvl_anom_score, pxl_lvl_anom_score

def process_image(heatmap: np.array, image: np.array):
    """与えられたheatmapをカラーマップに変換し、オリジナルのimageと結合して視覚化マップvisz_mapを作成"""
    #print(heatmap.shape, image.shape)  # (1,224,224) (152,900,3)
    heatmap = heatmap.transpose((1, 2, 0))  # チャンネル次元を最後に移動
    #print(heatmap.shape, image.shape)  # (224,224,1) (152,900,3)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    #print(heatmap.shape, image.shape)  # (152,900) (152,900,3)
    #print(heatmap)  # 0以上の実数
    #print(image) # 0以上の整数

    heatmap *= 5.  # ヒートマップ強調する。低いものは0.001とかになるので
    #heatmap = (heatmap - heatmap.min()) / heatmap.max() * 255
    heatmap = heatmap.astype(np.uint8)

    image = image.astype(np.uint8)

    heat_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # cv2.COLORMAP_JETが使用されており、出力はBGR形式
    visz_map = cv2.addWeighted(heat_map, 0.5, image, 0.5, 0)
    visz_map = cv2.cvtColor(visz_map, cv2.COLOR_BGR2RGB)
    visz_map = visz_map.astype(float)
    visz_map = visz_map / visz_map.max()
    return visz_map

def run_test(input_image: Image):
    """画像1枚推論"""
    # data
    image = input_image.convert("RGB")
    transforms = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
        ])
    image_transforms = transforms(image=np.array(image))["image"]
    # inference
    pred_score, pred_fmap = inference(image_transforms)
    # pred_fmapを画像に変換
    pred_fmap_image = process_image(pred_fmap, np.array(image))
    return pred_score, pred_fmap_image


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', type=str, default='',
                    help="出力ディレクトリのパス")
parser.add_argument('-m_d', '--model_dir', type=str, default='model',
                    help="モデルディレクトリのパス")
parser.add_argument('-m', '--model_cls', type=str, default="PatchCore(f_coreset=1.0, backbone_name='tf_efficientnetv2_b2.in1k')",
                    help="ind_knn_adのmodelクラス")
parser.add_argument('-s', '--seed', type=int, default=0,
                    help="乱数シード")
args = parser.parse_args()
if args.output_dir != "":
    os.makedirs(args.output_dir, exist_ok=True)
seed_everything(args.seed)

model = load_model(eval(args.model_cls), args.model_cls, args.model_dir)
model.eval().to(device)

demo = gr.Interface(
    fn=run_test,
    inputs=gr.Image(type="pil") ,
    outputs=[
        gr.Textbox(label="Prediction Score"),
        gr.Image(label="Anomaly Map")
    ],
    title="Anomaly Detection",  # インターフェースのタイトル
    allow_flagging='never', # 「フラグする」ボタンを表示しない
    live=True,  # 入力された画像に対して即座に処理を行う
)
demo.queue()  # タイムアウト対策  https://aotamasaki.hatenablog.com/entry/gradio-explanation
demo.launch(
    share=False,
    server_name="0.0.0.0",  # コンテナからの外部接続を受け入れるようにする
    server_port=7860,
    debug=True
)
