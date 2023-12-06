import argparse
import os
import sys
import gc
import glob
import random
import pickle
import traceback
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

sys.path.append('./ind_knn_ad/indad')
from models import SPADE, PaDiM, PatchCore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############
# Utils
##############
def get_score(labels: np.ndarray, predictions: np.ndarray):
    """auc計算"""
    # predictions が nan になったときの対策
    try:
        # soft_labelの場合でもいけるように閾値0.5で0, 1に変換
        labels = (labels > 0.5).astype(int)
        return roc_auc_score(labels, predictions)
    except:
        traceback.print_exc()
        return 0.0

def init_logger(log_file: str):
    """ロガー"""
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed: int):
    """乱数シード初期化"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def find_folders_with_name(directory: str, name: str = "fold"):
    """指定されたディレクトリ内で 'fold' を含むディレクトリを検索"""
    return [dir for dir in os.listdir(directory) if name in dir and os.path.isdir(os.path.join(directory, dir))]


##############
# Dataset
##############
def get_transforms():
    return A.Compose([
        #A.Resize(CFG.img_hw[0], CFG.img_hw[1]),
        #A.CenterCrop(CFG.img_hw[0], CFG.img_hw[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0),
        ToTensorV2(),
    ])

class MyDataset(Dataset):
    def __init__(self, train, transforms=None):
        self.train = train
        self.file_names = train['file_path'].values
        self.labels = train['label'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, index):
        img = cv2.imread(self.file_names[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.labels[index], dtype=torch.long)


##############
# Model
##############
def save_model(model: nn.Module, model_cls: str, output_dir: str):
    """ind_knn_adのモデル保存"""
    torch.save({'model': model.state_dict()}, f"{output_dir}/model.pth")
    # SPADEの推論はtrain setの特徴量マップと z_lib(マハラノビス距離？) が必要
    # https://nutritionfoodtech.com/2023/04/03/%e7%94%bb%e5%83%8f%e3%81%ae%e7%95%b0%e5%b8%b8%e6%a4%9c%e7%9f%a5-ind_knn_ad-%e3%82%92%e5%ad%a6%e7%bf%92%e3%81%a8%e6%8e%a8%e8%ab%96%e3%81%a7%e5%88%86%e5%89%b2%e3%81%99%e3%82%8b-%e3%80%90spade%e5%ad%a6/
    if "SPADE" in model_cls:
        pickle.dump(model.z_lib.cpu().detach(), open(f"{output_dir}/spade_z_lib.pkl", 'wb'))
        pickle.dump(model.feature_maps, open(f"{output_dir}/spade_feature_maps.pkl", 'wb'))  # list
    # PaDiMの推論は r_indices, means_reduced, E_inv, resize が必要
    elif "PaDiM" in model_cls:
        pickle.dump(model.r_indices.cpu().detach(), open(f"{output_dir}/padim_r_indices.pkl", 'wb'))
        pickle.dump(model.means_reduced.cpu().detach(), open(f"{output_dir}/padim_means_reduced.pkl", 'wb'))
        pickle.dump(model.E_inv.cpu().detach(), open(f"{output_dir}/padim_E_inv.pkl", 'wb'))  # E_invが異常に大きいファイルになる。8Gぐらい
        pickle.dump(model.resize.cpu(), open(f"{output_dir}/padim_resize.pkl", 'wb'))  # torch.nn.AdaptiveAvgPool2d
    # PatchCoreの推論は patch_lib, resize が必要
    elif "PatchCore" in model_cls:
        pickle.dump(model.patch_lib.cpu().detach(), open(f"{output_dir}/patchcore_patch_lib.pkl", 'wb'))
        pickle.dump(model.resize.cpu(), open(f"{output_dir}/patchcore_resize.pkl", 'wb'))  # torch.nn.AdaptiveAvgPool2d
        
def load_model(model: nn.Module, model_cls: str, output_dir: str):
    """ind_knn_adのモデルロード"""
    states = torch.load(f"{output_dir}/model.pth", map_location=torch.device("cpu"))
    load_flg = model.load_state_dict(states["model"])
    print("model load_flg:", load_flg, f"{output_dir}/model.pth")
    if "SPADE" in model_cls:
        model.z_lib = pickle.load(open(f"{output_dir}/spade_z_lib.pkl", 'rb'))
        model.feature_maps = pickle.load(open(f"{output_dir}/spade_feature_maps.pkl", 'rb'))
    elif "PaDiM" in model_cls:
        model.r_indices = pickle.load(open(f"{output_dir}/padim_r_indices.pkl", 'rb'))
        model.means_reduced = pickle.load(open(f"{output_dir}/padim_means_reduced.pkl", 'rb'))
        model.E_inv = pickle.load(open(f"{output_dir}/padim_E_inv.pkl", 'rb'))
        model.resize = pickle.load(open(f"{output_dir}/padim_resize.pkl", 'rb'))
    elif "PatchCore" in model_cls:
        model.patch_lib = pickle.load(open(f"{output_dir}/patchcore_patch_lib.pkl", 'rb'))
        model.resize = pickle.load(open(f"{output_dir}/patchcore_resize.pkl", 'rb'))
    return model
    
@torch.inference_mode()  # pytorch >= 1.9
def inference_loader(model: nn.Module, test_loader: DataLoader, device: str = device):
    """test_loaderでモデル推論"""
    model.eval().to(device)
    image_preds = []
    pixel_preds = []
    for images, _ in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            # ind_knn_ad
            # https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py#L65
            # score, segmentation_map = 異常度と特徴量マップ
            img_lvl_anom_score, pxl_lvl_anom_score = model.predict(images)    
        image_preds.append(img_lvl_anom_score.cpu().detach().numpy())
        pixel_preds.append(pxl_lvl_anom_score.cpu().detach().numpy())
    return np.stack(image_preds), np.stack(pixel_preds)  # shape=(N,), (N, 1, 224, 224)


##############
# Main
##############
def run_train_valid(args):
    """hold-out"""
    # data
    train = pd.read_csv(args.train_csv)
    valid = pd.read_csv(args.valid_csv)
    # model
    model = eval(args.model_cls).to(device)
    # train
    train_loader = DataLoader(MyDataset(train, transforms=get_transforms()))  # bs=1のDataLoader
    model.fit(train_loader)
    save_model(model, args.model_cls, args.output_dir)
    # inference
    valid_loader = DataLoader(MyDataset(valid, transforms=get_transforms()))  # bs=1のDataLoader. SPADEではbs増やすとエラーになった
    model = load_model(eval(args.model_cls), args.model_cls, args.output_dir)
    image_preds, _ = inference_loader(model, valid_loader)
    valid["pred"] = image_preds
    # score
    score = get_score(valid["label"].values, valid["pred"].values)
    LOGGER.info(f"valid auc: {score}")
    # output. 入力csvのファイル名に"_predict""を付けたcsvを出力
    output_csv = args.output_dir + f'/{Path(args.valid_csv).stem}_predict.csv'
    valid.to_csv(output_csv, index=False)
    print(f"=> OUTPUT: {output_csv}")

def run_cv(args):
    """cross-validation"""
    folds = pd.read_csv(args.train_csv)
    oof_df = pd.DataFrame()
    for fold in sorted( folds["fold"].unique() ):
        output_dir = args.output_dir + f"/fold{fold}"
        os.makedirs(output_dir, exist_ok=True)
        # data
        trn_idx = folds[folds["fold"] != fold].index
        val_idx = folds[folds["fold"] == fold].index
        train = folds.loc[trn_idx].reset_index(drop=True)
        valid = folds.loc[val_idx].reset_index(drop=True)
        train = train[train["label"] == 0].reset_index(drop=True)  # trainは正常データのみにする
        # model
        model = eval(args.model_cls).to(device)
        # train
        train_loader = DataLoader(MyDataset(train, transforms=get_transforms()))  # bs=1のDataLoader
        model.fit(train_loader)
        save_model(model, args.model_cls, output_dir)
        # inference
        valid_loader = DataLoader(MyDataset(valid, transforms=get_transforms()))  # bs=1のDataLoader. SPADEではbs増やすとエラーになった
        model = load_model(eval(args.model_cls), args.model_cls, output_dir)
        image_preds, _ = inference_loader(model, valid_loader)
        valid["pred"] = image_preds
        # fold score
        score = get_score(valid["label"].values, valid["pred"].values)
        LOGGER.info(f"fold{fold} auc: {score}")
        oof_df = pd.concat([oof_df, valid])
    # oof score
    score = get_score(oof_df["label"].values, oof_df["pred"].values)
    LOGGER.info(f"oof auc: {score}")
    # output. 入力csvのファイル名に"_predict""を付けたcsvを出力
    output_csv = args.output_dir + f'/{Path(args.train_csv).stem}_predict.csv'
    oof_df.to_csv(output_csv, index=False)
    print(f"=> OUTPUT: {output_csv}")

def run_test(args):
    """推論のみ"""
    # data
    test = pd.read_csv(args.test_csv)
    # inference
    test_loader = DataLoader(MyDataset(test, transforms=get_transforms()))  # bs=1のDataLoader. SPADEではbs増やすとエラーになった
    model = load_model(eval(args.model_cls), args.model_cls, args.output_dir)
    image_preds, _ = inference_loader(model, test_loader)
    test["pred"] = image_preds
    # output. 入力csvのファイル名に"_predict""を付けたcsvを出力
    output_csv = args.output_dir + f'/{Path(args.test_csv).stem}_predict.csv'
    test.to_csv(output_csv, index=False)
    print(f"=> OUTPUT: {output_csv}")

def run_test_folds(args):
    """全foldの推論平均"""
    # data
    test = pd.read_csv(args.test_csv)
    image_preds_sum = None
    fold_dirs = sorted( find_folders_with_name(args.output_dir, name="fold") )
    for fold_dir in fold_dirs:
        fold_dir = args.output_dir + f"/{fold_dir}"
        # inference
        test_loader = DataLoader(MyDataset(test, transforms=get_transforms()))  # bs=1のDataLoader. SPADEではbs増やすとエラーになった
        model = load_model(eval(args.model_cls), args.model_cls, fold_dir)
        image_preds, _ = inference_loader(model, test_loader)
        if image_preds_sum is None:
            image_preds_sum = image_preds
        else:
            image_preds_sum += image_preds
    test["pred"] = image_preds_sum / len(fold_dirs)
    # output. 入力csvのファイル名に"_predict""を付けたcsvを出力
    output_csv = args.output_dir + f'/{Path(args.test_csv).stem}_predict_folds.csv'
    test.to_csv(output_csv, index=False)
    print(f"=> OUTPUT: {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_csv', type=str, default='',
                        help="trainに使う正常画像のパス(file_path列)が書かれたcsv。trainのみ指定する場合はcross-validationを行う。fold列とlabel列が必要")
    parser.add_argument('-va', '--valid_csv', type=str, default='',
                        help="valid_setの画像パス(file_path列)とラベル(label列)が書かれたcsv。ラベルは0が正常、1が異常。valid指定する場合はhold-outを行う")
    parser.add_argument('-te', '--test_csv', type=str, default='',
                        help="推論したい画像のパス(file_path)が書かれたcsv")
    parser.add_argument('-o', '--output_dir', type=str, default='output',
                        help="出力ディレクトリのパス")
    parser.add_argument('-m', '--model_cls', type=str, default="PatchCore(f_coreset=.10, backbone_name='wide_resnet50_2')",
                        help="ind_knn_adのmodelクラス")
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help="乱数シード")
    parser.add_argument('--is_test_folds', action='store_true',
                        help="Test folds flag。全foldの推論平均するか")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    # train
    if (args.train_csv != "") and (args.valid_csv == ""):
        LOGGER = init_logger(args.output_dir + '/train.log')
        LOGGER.info(f"=> {args.model_cls}")
        run_cv(args)
    elif (args.train_csv != "") and (args.valid_csv != ""):
        LOGGER = init_logger(args.output_dir + '/train.log')
        LOGGER.info(f"=> {args.model_cls}")
        run_train_valid(args)
    # test
    if (args.test_csv != "") and args.is_test_folds:
        run_test_folds(args)
    elif args.test_csv != "":
        run_test(args)

