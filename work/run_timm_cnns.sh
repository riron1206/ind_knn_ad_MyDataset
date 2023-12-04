# ファイルから配列を読み込む。使用可能なtimmのbackbone_name
# ※ViT系のモデルは使えない。timmのfeatures_onlyの引数が使えないため
readarray -t array < timm_cnns.txt
for i in "${array[@]}"
do
    echo $i
    python run_mydataset.py \
    -tr ./train.csv \
    -va ./valid.csv \
    -o ./output/run_mydataset/sample_data \
    -m "PatchCore(f_coreset=0.1, backbone_name=$i, coreset_eps=0.9)"
done
