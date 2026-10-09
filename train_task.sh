set mp3d path
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

export CUDA_VISIBLE_DEVICES=1,2,3

# Task1
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi1.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task1 \
    --TaskID Task1 \

# Task2
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi2.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task2 \
    --TaskID Task2 \

# Task3
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi3.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task3 \
    --TaskID Task3 \

# Task4
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi4.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task4 \
    --TaskID Task4 \
# Task5
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi5.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task5 \
    --TaskID Task5 \
# Task 6
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi6.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --test_datasets CVDN \
    --max_saved_checkpoints 1 \
    --output_dir output/Task6 \
    --TaskID Task6 \
# Task7
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi7.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-4 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task7 \
    --TaskID Task7 \
# Task8
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi8.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task8 \
    --TaskID Task8 \
# Task9
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi9.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task9 \
    --TaskID Task9 \
# Task10
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi10.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task10 \
    --TaskID Task10 \
# Task11
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi11.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task11 \
    --TaskID Task11 \

# Task12
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi12.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --test_datasets CVDN \
    --max_saved_checkpoints 1 \
    --output_dir output/Task12 \
    --TaskID Task12 \
# Task13
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi13.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task13 \
    --TaskID Task13 \

# Task14
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi14.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task14 \
    --TaskID Task14 \
# Task15
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi15.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task15 \
    --TaskID Task15 \

# Task16
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi16.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_fgr2r \
    --test_datasets R2R \
    --max_saved_checkpoints 1 \
    --output_dir output/Task16 \
    --TaskID Task16 \

# Task17
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi17.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --teacher_forcing_coef 1 \
    --enable_og \
    --test_datasets REVERIE \
    --max_saved_checkpoints 1 \
    --output_dir output/Task17 \
    --TaskID Task17 \

# Task18
torchrun --nnodes=1 --nproc_per_node=3 --master_port 41000 train_new.py \
    --stage multi \
    --cfg_file configs/multi18.yaml \
    --data_dir data \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --batch_size 1 \
    --gradient_accumulation_step 8 \
    --lr 1e-3 \
    --seed 3407 \
    --num_epochs 500 \
    --test_datasets CVDN \
    --max_saved_checkpoints 1 \
    --output_dir output/Task18 \
    --TaskID Task18