source=Breast
target=Bone
datadir=datasets
crop_size=256
augment_probability=0.8
generator=fcngen
discriminator=patchgan
detector=micronet
n_epochs=100
batch_size=1
gpu=1
result=learned_models

pretrain_weights=PATH_TO_SOURCE_MODEL
pretrain_weights_a=PATH_TO_SOURCE_MODEL
outdir=${result}/${source}2${target}
python train.py ${outdir} \
    --datadir ${datadir} \
    --source ${source} \
    --target ${target} \
    --crop_size ${crop_size} \
    --generator ${generator} \
    --discriminator ${discriminator} \
    --detector ${detector} \
    --n_epochs ${n_epochs} \
    --batch_size ${batch_size} \
    --gpu ${gpu} \
    --pretrain_weights ${pretrain_weights} \
    --pretrain_weights_a ${pretrain_weights_a} \
    --use_crossdomain \
    --augment_probability ${augment_probability}
