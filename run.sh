#!/bin/bash
# 指定要使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1

python main.py --dataset SYSU-CD --model_type cd --model_arch SEED_PEFT --peft_method lora --model_name DINO3CD --exp_name DINO3CD_DPT_SYSU_lora --max_steps 15000 --batch_size 16 --devices 2 --strategy ddp_find_unused_parameters_true --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs

python main.py --dataset SYSU-CD --model_type cd --model_arch SEED_PEFT --peft_method adapter --model_name DINO3CD --exp_name DINO3CD_DPT_SYSU_adapter --max_steps 15000 --batch_size 16 --devices 2 --strategy ddp_find_unused_parameters_true --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs


python main.py --dataset SYSU-CD --model_type cd --model_arch SEED_PEFT --peft_method lora --model_name SAM2CD --exp_name SAM2CD_DPT_SYSU_lora --max_steps 15000 --batch_size 16 --devices 2 --strategy ddp_find_unused_parameters_true --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs

python main.py --dataset SYSU-CD --model_type cd --model_arch SEED_PEFT --peft_method adapter --model_name SAM2CD --exp_name SAM2CD_DPT_SYSU_adapter --max_steps 15000 --batch_size 16 --devices 2 --strategy ddp_find_unused_parameters_true --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs