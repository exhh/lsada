source=Breast
datadir=datasets
crop_size=256
detector=micronet
n_epochs=500
batch_size=4
gpu=1
result=learned_models

outdir=${result}/${source}
python train_source.py ${outdir} \
    --datadir ${datadir} \
    --source ${source} \
    --crop_size ${crop_size} \
    --detector ${detector} \
    --n_epochs ${n_epochs} \
    --batch_size ${batch_size} \
    --gpu ${gpu} \
    --use_validation
