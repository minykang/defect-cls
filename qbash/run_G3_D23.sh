#!/bin/bash

# 여러 모델을 실험하기 위해 배열에 모델 이름을 저장합니다.
model_names=("tf_efficientnetv2_m.in21k" "tf_efficientnetv2_l.in21k")

# 여러 데이터셋을 실험하기 위해 배열에 전략을 저장합니다.
strats=("3")

# 스크립트의 다른 변수들을 정의합니다.
version="v3"

# 각 데이터셋과 각 모델에 대해 반복하면서 main.py를 실행합니다.
for Strat in "${strats[@]}"
do
    for Model_name in "${model_names[@]}"
    do
        echo "실험 중인 모델: ${Model_name}, 전략: ${Strat}"
        CUDA_VISIBLE_DEVICES=3 python main.py \
        --model_name ${Model_name} \
        --version ${version} \
        --strat ${Strat}
    done
done
