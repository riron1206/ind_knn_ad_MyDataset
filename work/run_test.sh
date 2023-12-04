# sample_data
python run_mydataset.py \
-tr ./train.csv \
-va ./valid.csv \
-o ./output/sample_data \
-m "PatchCore(f_coreset=.10, backbone_name='wide_resnet50_2')"

python run_mydataset.py \
-te ./valid.csv \
-o ./output/sample_data \
-m "PatchCore(f_coreset=.10, backbone_name='wide_resnet50_2')"


# PatchCore params search
array=(".10" ".20" ".30" ".40" ".50" ".60" ".70" ".80" ".90" "1.0")
for i in "${array[@]}"
do
    # sample_data
    python run_mydataset.py \
    -tr ./train.csv \
    -va ./valid.csv \
    -o ./output/run_mydataset/sample_data \
    -m "PatchCore(f_coreset=$i, backbone_name='wide_resnet50_2')"
done

array=(".10" ".20" ".30" ".40" ".50" ".60" ".70" ".80" ".90" "1.0")
for i in "${array[@]}"
do
    # sample_data
    python run_mydataset.py \
    -tr ./train.csv \
    -va ./valid.csv \
    -o ./output/run_mydataset/sample_data \
    -m "PatchCore(f_coreset=0.1, backbone_name='wide_resnet50_2', coreset_eps=$i)"
done
