# ind_knn_ad_MyDataset

KNN(k-nearest neighbors)ベースの画像異常検知のgithubである[ind_knn_ad](https://github.com/rvorias/ind_knn_ad/tree/master)のサンプルコード。自前のDatasetで実行できるようにした。

## Prerequisites

- Dockerが必要。インストール方法は[公式ドキュメント](https://docs.docker.com/get-docker/)を参照。

- nvidia-driverが必要。インストール方法は[この記事](https://qiita.com/y-vectorfield/items/72bfb66d8ec85847fe2f)を参考にした。

## Build environment

```bash
# Docker image作成
docker build -t ind_knn_ad -f ./Dockerfile .

# Dockerコンテナ起動してbashで入る
docker run -p 8888:8888 \
-it \
-w /work \
-v $PWD/work:/work \
--rm \
--gpus all \
--shm-size 32g \
ind_knn_ad \
/bin/bash

# ind_knn_adのコードダウンロード
git clone https://github.com/rvorias/ind_knn_ad.git
```

## Run

[./work/run_mydataset.py](./work/run_mydataset.py) がサンプルコード。使用例は[./work/run_test.sh](./work/run_test.sh) を参照。

- timmをbackboneに用いて [SPADE](./images/ind_knn_ad_methods.md), [PaDiM](./images/ind_knn_ad_methods.md), [PatchCore](./images/ind_knn_ad_methods.md) の3つの異常検知手法が使える。

- 入力画像のファイルパスが記載されたCSVファイルを入力として使用。

[./work/run_timm_cnns.sh](./work/run_timm_cnns.sh) は色んなbackboneで実行する例。

## Jupyter notebook

以下のコマンドでJupyter labを起動可能。

```bash
# Dockerコンテナ起動してbashで入る
docker run -p 8888:8888 \
-it \
-w /work \
-v $PWD/work:/work \
--rm \
--gpus all \
--shm-size 32g \
ind_knn_ad \
/bin/bash

# Jupyter lab起動
jupyter lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --port=8888

# ind_knn_adを試したnotebook: work/ind_knn_ad_MyDataset.ipynb が実行できる


```
