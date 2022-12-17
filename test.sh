source=Breast
target=Bone
datadir=datasets
detector=micronet
checkpoint=PATH_TO_LEARNED_MODEL
gpu=2
split_test=test
outdir=experiments

model_path=learned_models/${source}2${target}
python test.py ${outdir} \
    --train_val_test ${split_test} \
    --datadir ${datadir} \
    --target ${target} \
    --detector ${detector} \
    --model_path ${model_path} \
    --checkpoint ${checkpoint} \
    --gpu ${gpu}
