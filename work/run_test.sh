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
