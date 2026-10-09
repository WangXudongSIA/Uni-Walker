import os
import json
import torch
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
from tools.common_utils import all_gather
from tools.parser import read_args, random_seed
from tasks.loaders import create_dataloaders
from tasks.feature_db import create_feature_db, create_object_feature_db
# from models.nav_model import NavModel
from models.nav_model import *
from tools.optims import dist_models, save_checkpoint
from tools.trie import Trie
from transformers import AutoModelForCausalLM
import loralib as lora
from loralib import Linear as LoRALinear
from loralib import LoRALayer
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.autograd as autograd

class SSCRegularizer:
    def __init__(self, A_param, lambda_ssc=0.1, omega=0.9, 
                 prev_F_A=None, device="cuda"):

        self.A = A_param
        self.lambda_ssc = lambda_ssc
        self.omega = omega
        self.device = device

        if prev_F_A is not None:
            self.F_A = prev_F_A.clone().to(self.device)
        else:
            self.F_A = torch.zeros_like(self.A, device=self.device)

        self.A_prev = self.A.detach().clone()

        self.F_A_current = torch.zeros_like(self.A, device=self.device)

    @torch.no_grad()
    def update_Fisher(self, model, dataloader, loss_fn, num_batches=None):

        fisher_estimate = torch.zeros_like(self.A, device=self.device)

        for i, (x, y) in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)

            logits = model(x)
            loss = loss_fn(logits, y)

            grads = autograd.grad(loss, self.A, retain_graph=False, create_graph=False)[0]
            fisher_estimate += grads.detach() ** 2

        fisher_estimate /= (i + 1)

        self.F_A_current = fisher_estimate.clone()

        self.F_A = self.omega * self.F_A + (1 - self.omega) * fisher_estimate

        self.A_prev = self.A.detach().clone()

    def compute_ssc_loss(self):

        diff = self.A - self.A_prev
        ssc_loss = self.lambda_ssc * torch.sum(self.F_A * (diff ** 2))
        return ssc_loss

    def get_current_task_Fisher(self):

        return self.F_A_current.detach().clone()

    def get_global_Fisher(self):

        return self.F_A.detach().clone()

    def get_prev_A(self):

        return self.A_prev.detach().clone()


def lora_and_H_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def mark_only_loraB2_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():

        p.requires_grad = False

    for n, p in model.named_parameters():
        if 'lora_B2' in n:
            p.requires_grad = True
        elif 'lora_B1' in n:
            p.requires_grad = False
        elif 'lora_A' in n:
            p.requires_grad = True

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def mark_only_lora_and_H_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

class DualBSharedALoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:

            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))

            self.lora_B1 = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_B2 = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.zeros_(self.lora_B2)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    total_B = self.lora_B1 + self.lora_B2
                    self.weight.data -= T(total_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    total_B = self.lora_B1 + self.lora_B2
                    self.weight.data += T(total_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            lora_out = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
            LoRA_result = (lora_out @ (self.lora_B1 + self.lora_B2).transpose(0, 1)) * self.scaling
            return result + LoRA_result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class BFloat16DualBSharedALoRALinear(DualBSharedALoRALinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(self.weight.to(torch.bfloat16))
        self.lora_A = torch.nn.Parameter(self.lora_A.to(torch.bfloat16))
        self.lora_B1 = torch.nn.Parameter(self.lora_B1.to(torch.bfloat16))
        self.lora_B2 = torch.nn.Parameter(self.lora_B2.to(torch.bfloat16))
        
class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        if self.num == 0:
            return 0
        return self.total / self.num


def train_one_epoch(
        args,
        global_cfg,
        model,
        optimizer,
        lr_scheduler,
        criterion,
        dataloaders,
        agents,
        epoch,
        logger,
        stage='multi'
):

    model.train()
    entropy_metric = Metrics()
    loss_metric = Metrics()
    instr_pred_metric = Metrics()

    num_batches_per_epoch = dataloaders.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        range(dataloaders.num_batches),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    
    dataset_cfg = global_cfg.Pretrain if stage=='pretrain' else global_cfg.Multi
    loss_stats = {k: Metrics() for k in dataset_cfg.SOURCE}

    for step, (name, batch) in enumerate(dataloaders):
        loss_coef = dataset_cfg.LOSS_COEF.get(name, 1.)
        # perform embodied tasks
        # the actual batch_size equals to args.batch_size * world_size * (args.gradient_accumulation_step)
        dataset = dataloaders.loader.get_dataset(name)
        agent = agents.get(name)
        loss = agent.train(
            name,
            batch,
            args,
            global_cfg,
            model=model,
            criterion=criterion,
            dataset=dataset,
            step=step,
            entropy_metric=entropy_metric,
            instr_pred_metric=instr_pred_metric
        )
        loss_metric.accumulate(loss.item())
        loss_stats[name].accumulate(loss.item())

        if (step+1) % args.gradient_accumulation_step==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.)
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        if args.rank == 0:
            verbose_dict = dict(
                step=step,
                name=name,
                # index=batch['sample_idx'],
                loss=loss_metric.average,
                entropy=entropy_metric.average,
                instr_pred_metric=instr_pred_metric.average,
                lr=lr_scheduler.get_last_lr()[0],
            )
            for k in dataset_cfg.SOURCE:
                verbose_dict[k] = loss_stats[k].average
            pbar.set_postfix(verbose_dict)
            pbar.update()

        if step == num_batches_per_epoch-1:
            logger.info("***** train [{}] epoch *****".format(epoch))
            train_stat_str = 'Loss: %.2f\n' % loss_metric.average
            train_stat_str += "Instr_pred: %.2f\n" % instr_pred_metric.average
            for task in dataset_cfg.SOURCE:
                train_stat_str += "%s: %.2f\n" % (task, loss_stats[task].average)
            logger.info(train_stat_str)
            break

@torch.no_grad()
def val_one_epoch(
        args,
        global_cfg,
        model,
        dataloaders,
        agents,
        epoch,
        logger,
) -> Dict[str, Dict[str, float]]:

    model.eval()
    entropy_metric = Metrics()

    loss_str = "\n[Eval] {} epoch {}\n".format(args.validation_split, epoch)
    task_results = {}
    for name, loader in dataloaders.items():        
        logger.info("***** validate {} split on {} task *****".format(args.validation_split, name))
        dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        preds = agent.validate(
            name,
            args,
            global_cfg,
            model,
            loader,
            entropy_metric=entropy_metric
        )

        all_preds = all_gather(preds)
        all_preds = merge_dist_results(all_preds)

        if args.rank == 0 and not args.validation_split.startswith('test'):
            score_summary, item_metrics = dataset.eval_metrics(all_preds, logger=logger, name=name)

            task_results[name] = score_summary
            loss_str += "\n [Eval] dataset=[{}] \n".format(name)
            for metric, val in score_summary.items():
                if metric == 'sr':
                    loss_str += '\n[Eval] ||| %s: %.2f' % (metric, val)
                else:
                    loss_str += ', %s: %.2f' % (metric, val)
        
        if args.rank== 0 and args.save_pred_results:
            dataset.save_json(
                all_preds, 
                os.path.join(args.output_dir, f"{name}_{args.validation_split}.json"),
                item_metrics=item_metrics if args.save_detail_results else None
            )

    logger.info(loss_str)
    
    return task_results

def find_most_similar_historical_task(current_task_id: int, feature_dim: int = 1028) -> int:

    current_path = f'output/Task{current_task_id}/feature.npy'
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"{current_path} No")
    
    current_feature = np.load(current_path).reshape(-1, feature_dim)
    current_mean = np.mean(current_feature, axis=0, keepdims=True)

    max_similarity = -1.0
    most_similar_task_id = -1

    for prev_id in range(1, current_task_id):
        prev_path = f'output/Task{prev_id}/feature.npy'
        if not os.path.exists(prev_path):
            continue
        try:
            prev_feature = np.load(prev_path).reshape(-1, feature_dim)
            prev_mean = np.mean(prev_feature, axis=0, keepdims=True)
            sim = cosine_similarity(current_mean, prev_mean)[0][0]
        except Exception as e:
            continue

        if sim > max_similarity:
            max_similarity = sim
            most_similar_task_id = prev_id

    return most_similar_task_id

@torch.no_grad()
def LoRA_Aggregation(
        device,
        args,
        global_cfg,
        model,
        dataloaders,
        agents,
) -> Dict[str, Dict[str, float]]:

    model.eval()
    entropy_metric = Metrics()
    for name, loader in dataloaders.items():        
        dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        Room_feature, Room_instruction = agent.LoRAID_validate(
            name,
            args,
            global_cfg,
            model,
            loader,
            entropy_metric=entropy_metric
        )
        
    feature = Room_feature[0][0].reshape(1, -1)
    similarity_results = []
    for i in range(1, 19):
        task_name = f'Task{i}'
        path = f'output/{task_name}/feature.npy'
        scene_feature = np.load(path) 
        scene_feature_flat = scene_feature.reshape(-1, 1028)
        sims = cosine_similarity(feature, scene_feature_flat) 
        max_sim = np.max(sims)
        max_sim = max_sim ** 10
        similarity_results.append(max_sim)
        
    arr = np.array(similarity_results)
    arr = arr - np.mean(arr)
    arr[arr < 0.01] = 0
    normalized = arr / np.sum(arr)
    print(normalized)
    weight = normalized

    if np.max(normalized) < 0.5:
        similarity_results = []
        for i in range(1, 19):
            task_name = f'Task{i}'
            path = f'output/{task_name}/text.npy'
            text_feature = np.load(path)
            text_feature_flat = text_feature.reshape(-1, 768)
            max_sim = 0 
            for text_feat in Room_instruction:
                feature = text_feat[0].reshape(1, -1)
                sims = cosine_similarity(feature, text_feature_flat)
                max_sim = max(max_sim, np.max(sims))
            max_sim = max_sim
            similarity_results.append(max_sim)

        if np.isnan(similarity_results).any():
            weight = normalized
        else:
            arr = np.array(similarity_results)
            top3_indices = np.argpartition(arr, -3)[-3:]
            binary_arr = np.zeros_like(arr)
            binary_arr[top3_indices] = 1
            normalized = normalized * binary_arr
            weight = normalized / np.sum(normalized)
            print(weight)

    max_index = np.argmax(weight)
    task_name = f'Task{max_index + 1}'
    path = os.path.join('output', task_name, 'LoRA.pt')
    state_dict = torch.load(path, map_location=device)
    lora_weights = {k.replace('module.', ''): v for k, v in state_dict.items()}
    loaded, missing = model.load_state_dict(lora_weights, strict=False)
    print(f"LoRA loaded from {task_name}: {len(loaded)} keys, missing: {len(missing)}")
    
    return model

@torch.no_grad()
def Save_Room_feature(
        feature_path, 
        text_path,
        args,
        global_cfg,
        model,
        dataloaders,
        agents,
        epoch,
        logger,
) -> Dict[str, Dict[str, float]]:

    model.eval()
    entropy_metric = Metrics()
    for name, loader in dataloaders.items(): 
    # for step,(name, loader) in enumerate(dataloaders):     
        # dataset = dataloaders[name].get_dataset()
        agent = agents[name]
        Room_feature, Room_instruction = agent.LoRAID_validate(
            name,
            args,
            global_cfg,
            model,
            loader,
            entropy_metric=entropy_metric
        )
        
    np.save(feature_path, Room_feature)
    np.save(text_path, Room_instruction)
    

def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs

def calc_overall_score(results, cfg):
    score = 0.
    for task in results:
        if task not in cfg.Multi.SOURCE:
            continue
        if task == 'R2R':
            score += results[task]['sr'] / 70
            # score += results[task]['spl'] / 60
        elif task == 'REVERIE':
            score += results[task]['sr'] / 40
            # score += results[task]['spl'] / 36.63
        elif task == 'CVDN':
            score += results[task]['sr'] / 30
            # pass
        elif task == 'SOON':
            score += results[task]['spl'] / 26.58
        elif task == 'EQA':
            pass
        elif task == "ScanQA":
            pass
        else:
            raise NotImplementedError(f"The method for calculating the score of {task} is not Implemented.")

    return score

def main():
    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed + args.rank)

    ##################### DATASET #####################
    feat_db = create_feature_db(global_cfg.Feature.feature_database, global_cfg.Feature.image_feat_size, args)
    obj_feat_db = create_object_feature_db(global_cfg.Feature.object_database, global_cfg.Feature.obj_feat_size, args)
    # Initialize train dataloader
    if args.mode == "train":
        train_dataloaders, train_agents = create_dataloaders(
            args, global_cfg, logger,
            training=True, device=device_id, feat_db=feat_db, obj_feat_db=obj_feat_db, stage=args.stage
        )
    # Initialize val dataloader
    val_dataloaders, val_agents = create_dataloaders(
        args, global_cfg, logger,
        training=False, device=device_id, feat_db=feat_db, obj_feat_db=obj_feat_db, stage="multi"
    )

    # Model
    model = NavModel(args, logger, global_cfg.Model)
    
    if args.mode=="train":
        for param in model.parameters():
            param.requires_grad = False
        for layer in model.lang_model.model.layers:

            layer.self_attn.q_proj = BFloat16DualBSharedALoRALinear(
                in_features=layer.self_attn.q_proj.in_features,
                out_features=layer.self_attn.q_proj.out_features,
                r=8,
                lora_alpha=16,
                merge_weights=False
            )
            layer.self_attn.v_proj = BFloat16DualBSharedALoRALinear(
                in_features=layer.self_attn.v_proj.in_features,
                out_features=layer.self_attn.v_proj.out_features,
                r=8,
                lora_alpha=16,
                merge_weights=False
            )

        TaskID = args.TaskID
        prefix = ''.join(filter(str.isalpha, TaskID))
        number = int(''.join(filter(str.isdigit, TaskID)))
        new_TaskID = f"{prefix}{number - 1}"    
        print("-------------------------------------------", new_TaskID)
        if number >= 2:
            epoch = 1
            feature_path = Path(args.output_dir) / f"feature.npy"
            text_path = Path(args.output_dir) / f"text.npy"
            Save_Room_feature(feature_path, text_path, args, global_cfg, model, val_dataloaders, val_agents, epoch, logger)
            
            Load_LoRA_path = "output/" + new_TaskID + "/LoRA.pt"
            Last_lora_weights = torch.load(Load_LoRA_path)
            Last_lora_weights = {k.replace('module.', ''): v for k, v in Last_lora_weights.items()}
            Last_lora_weights = {k: v for k, v in Last_lora_weights.items() if 'lora_A' in k}
            loaded, missing = model.load_state_dict(Last_lora_weights, strict=False) 

            TOP_TaskID = find_most_similar_historical_task(number)
            Load_LoRA_TOP_path = "output/Task" + str(TOP_TaskID) + "/LoRA.pt"
            Last_lora_weights = torch.load(Load_LoRA_TOP_path)
            Last_lora_weights = {k.replace('module.', ''): v for k, v in Last_lora_weights.items()}
            Last_lora_weightsB = {k: v for k, v in Last_lora_weights.items() if 'lora_B' in k}
            loaded, missing = model.load_state_dict(Last_lora_weightsB, strict=False) 

        if number >= 2:
            mark_only_loraB2_trainable(model)
        else:
            mark_only_lora_and_H_as_trainable(model)
        
    if args.mode=="test":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for layer in model.lang_model.model.layers: 
            layer.self_attn.q_proj = BFloat16DualBSharedALoRALinear(
                in_features=layer.self_attn.q_proj.in_features,
                out_features=layer.self_attn.q_proj.out_features,
                r=8,         
                lora_alpha=16,     
                merge_weights=False 
            )
            layer.self_attn.v_proj = BFloat16DualBSharedALoRALinear(
                in_features=layer.self_attn.v_proj.in_features,
                out_features=layer.self_attn.v_proj.out_features,
                r=8,
                lora_alpha=16,  
                merge_weights=False
            )
            
        epoch = 1
        model = LoRA_Aggregation(
            device, args, global_cfg, model, val_dataloaders, val_agents, epoch, logger
        )
        
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')
    
    model, optimizer, resume_from_epoch, lr_scheduler = dist_models(args, model, logger)
        
    if args.mode=="test":
        logger.info("**************************** Test ****************************")
        model.eval()
        epoch = 1

        # feature_path = Path(args.output_dir) / f"feature.npy"
        # text_path = Path(args.output_dir) / f"text.npy"
        # Save_Room_feature(feature_path, text_path, args, global_cfg, model, val_dataloaders, val_agents, epoch, logger)

        
        results = val_one_epoch(
            args, global_cfg, model, val_dataloaders, val_agents, epoch, logger
        )
        
    elif args.mode == "train":
        logger.info("**************************** Train ****************************")
        epoch = 1
        feature_path = Path(args.output_dir) / f"feature.npy"
        text_path = Path(args.output_dir) / f"text.npy"
        Save_Room_feature(feature_path, text_path, args, global_cfg, model, val_dataloaders, val_agents, epoch, logger)
        best_results, best_score = None, None
        history_scores = []

        # if TaskID > 1:
        #     Fisher_checkpoint = torch.load("task_fisher.pt")
        #     Fisher_loaded = Fisher_checkpoint["Fisher"]
        #     ssc_reg = SSCRegularizer(model.lang_model.model.layers.lora_A, lambda_ssc=0.1, omega=0.9, 
        #                   prev_F_A=Fisher_loaded)
        # else:
        #     ssc_reg = SSCRegularizer(model.A, lambda_ssc=0.1, omega=0.9)

        for epoch in range(resume_from_epoch, args.num_epochs):
            # training
            train_one_epoch(
                args, global_cfg, model, optimizer, lr_scheduler, criterion, train_dataloaders, train_agents, epoch, logger, stage=args.stage
            )

            # evaluation
            results = val_one_epoch(
                args, global_cfg, model, val_dataloaders, val_agents, epoch, logger
            )
            
            score = calc_overall_score(results, global_cfg)
            history_scores.append(score)
            should_save_checkpoint = False
    
            if best_results is None or score >= best_score:
                    best_results = results
                    best_score = score
                    should_save_checkpoint = True
                    
            logger.info(f"Current Score: {score}")
            logger.info(f"Best Score: {best_score}")
            
            if should_save_checkpoint:
                model_path = Path(args.output_dir) / f"LoRA.pt"
                torch.save(lora_and_H_state_dict(model), model_path)
                # torch.save(lora.lora_state_dict(model), model_path)
                
        
        # print best results
        logger.info(f"Best Results:")
        logger.info(best_results)
                    

if __name__ == '__main__':
    main()
