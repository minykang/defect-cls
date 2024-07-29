#!/bin/bash

# 여러 모델을 실험하기 위해 배열에 모델 이름을 저장합니다.

model_names=("resnet50.tv_in1k" "resnet101.tv_in1k" "tf_efficientnetv2_m.in21k" "tf_efficientnetv2_l.in21k")

# 여러 데이터셋을 실험하기 위해 배열에 전략을 저장합니다.
# strats=("1" "3" "5" "10" "new1" "new3" "new5" "new10")
strats=("5"  "new5")

# 스크립트의 다른 변수들을 정의합니다.
version="v4"

# 각 데이터셋과 각 모델에 대해 반복하면서 main.py를 실행합니다.
for Strat in "${strats[@]}"
do
    for Model_name in "${model_names[@]}"
    do
        echo "실험 중인 모델: ${Model_name}, 전략: ${Strat}"
        CUDA_VISIBLE_DEVICES=2 python OPTIM-SGD/main_v2.py \
        --model_name ${Model_name} \
        --version ${version} \
        --strat ${Strat} \
        --lr 0.008 \
        --n_epochs 100
    done
done




