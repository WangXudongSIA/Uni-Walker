
# Task1
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi1.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 1 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task2
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi2.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task3
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi3.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

# Task4
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi4.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task5
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi5.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

# Task 6
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi6.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets CVDN \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task7
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi7.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task8
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi8.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

# Task9
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi9.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task10
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi10.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

# Task11
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi11.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task12
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi12.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets CVDN \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task13
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi13.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

# Task14
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi14.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task15
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi15.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

# Task16
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi16.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets R2R \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \

# Task17
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi17.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets REVERIE \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results \
    --do_sample \
    --temperature 0.01 \
    --enable_og \

Task18
python train_new.py --stage multi \
    --mode test \
    --data_dir data \
    --cfg_file configs/multi18.yaml \
    --pretrained_model_name_or_path data/models/Vicuna-7B \
    --precision amp_bf16 \
    --resume_from_checkpoint model_with_pretrain.pt \
    --test_datasets CVDN \
    --val_batch_size 4 \
    --output_dir build/eval \
    --validation_split val_seen \
    --save_pred_results