#!/bin/bash
# get all filename in specified path
 
path=/data/YIXIAOKAI/20190710/dataset/TEST_RA/
name=wn11_4_0_1
files=$(ls $path)
for filename in $files
do
 CUDA_VISIBLE_DEVICES=0 python3 main.py --model wn --chop --dir_demo ../../../dataset/TEST_RA/$filename/QP22 --data_test Demo --scale 1 --save ../../../latest/RA/$name/$filename --pre_train ../experiment/epoch_50/model/model_latest.pt --test_only --save_results 
 CUDA_VISIBLE_DEVICES=0 python3 main.py --model wn --chop --dir_demo ../../../dataset/TEST_RA/$filename/QP27 --data_test Demo --scale 1 --save ../../../latest/RA/$name/$filename --pre_train ../experiment/epoch_50/model/model_latest.pt --test_only --save_results 
 CUDA_VISIBLE_DEVICES=0 python3 main.py --model wn --chop --dir_demo ../../../dataset/TEST_RA/$filename/QP32 --data_test Demo --scale 1 --save ../../../latest/RA/$name/$filename --pre_train ../experiment/epoch_50/model/model_latest.pt --test_only --save_results
 CUDA_VISIBLE_DEVICES=0 python3 main.py --model wn --chop --dir_demo ../../../dataset/TEST_RA/$filename/QP37 --data_test Demo --scale 1 --save ../../../latest/RA/$name/$filename --pre_train ../experiment/epoch_50/model/model_latest.pt --test_only --save_results 


done



