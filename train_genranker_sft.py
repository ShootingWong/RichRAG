import argparse
import random
import numpy as np
import torch.nn as nn
import torch
import torch.nn.utils as utils
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
import transformers
from tqdm import tqdm
import os
import wandb
from dataclasses import dataclass, field
from typing import Optional
import json
from safetensors import safe_open

from src.dataset_sft_rank import MyDataset
from src.ranker_sft import Ranker
from src.gen_ranker import GenRanker
from src.diversity_metric_raw import find_doc_scores, get_next_doc_score

import loguru 

os.environ['WANDB_API_KEY'] = '{your_wandb_key}'
def cast_to_precision(model, precision):
    if precision == "fp32":
        return model
    elif precision == "fp16":
        model.to(torch.float16)
    elif precision == "bf16":
        model.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported precision {precision}, must be one of fp32, fp16, bf16")
    return model

device = torch.device("cuda:0")
torch.autograd.set_detect_anomaly(True)

MAX_CANDI = 0

@dataclass
class ModelArguments:
    rank_model_path: Optional[str] = field(default="")
    pretrain_model_path: Optional[str] = field(default="")
    model_type: Optional[str] = field(default="graph")
    loss_type: Optional[str] = field(default="pair")
    load_model: Optional[int] = field(default=0)
    load_path: Optional[str] = field(default="")
    save_path: Optional[str] = field(default="")
    seeed: int = field(default=0)
    is_training: Optional[int] = field(default=1)
    rpc: Optional[int] = field(default=0)
    topk: int = field(default=10)
    max_subq_cnt: int = field(default=7)
    special_token_format: str = field(default="[D{}]")
    special_query_token: str = field(default="[Q]")
    special_sub_token: str = field(default="[SUBQ]")
    special_split_token: str = field(default="[SEP]")
    tmp: float = field(default=1)
    q_maxlength: int = field(default=25)
    d_maxlength: int = field(default=300)
    load_step: int  = field(default=0)
    max_subq_cnt_per: int = field(default=6)
    
    max_candi_cnt: int = field(default=528)
    l2: float = field(default=1.0)
    distrub: int = field(default=0)
    
    wandb_project: Optional[str] = field(default="SFT-Ranker")
    wandb_name: Optional[str] = field(default="1")
    
    test: int = field(default=0)
    efficient: int = field(default=0)

@dataclass
class DataArguments:
    train_data: str = field(default="../data/", metadata={"help": "Path to the data."})
    eval_data: str = field(default="../data/", metadata={"help": "Path to the data."})
    test_data: str = field(default="../data/", metadata={"help": "Path to the data."})
    train_retrieve_path: str = field(default="../data/", metadata={"help": "Path to the retrieve result."})
    eval_retrieve_path: str = field(default="../data/", metadata={"help": "Path to the retrieve result."})
    test_retrieve_path: str = field(default="../data/", metadata={"help": "Path to the retrieve result."})
    task: str = field(default="nq")
    window: int = field(default=2)
    
    
def load_safetensor_state_dict(path):
    tensors = {}
    loguru.logger.info(f'LOAD PATH = {path}')
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
            
def set_seed(seed=3704):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_psgs_subs(psgs):
    doc_str_list = []
    all_subqs = []
    for sub_p in psgs:
        subqs = sub_p['subqs']
        p = sub_p['doc']
        title = p['title']
        content = p['text']
        if "section" in p and len(p["section"]) > 0:
            title = f"{p['title']}: {p['section']}"

        doc_str = f'{title} {content}'
        doc_str_list.append(doc_str)
        all_subqs.append(subqs)

    return doc_str_list, all_subqs
    
def train_model(model_args, data_args, training_args):

    loguru.logger.info(f'args.rank_model_path = {model_args.rank_model_path}')
    
    config = AutoConfig.from_pretrained(model_args.rank_model_path)
    rank_tokenizer = AutoTokenizer.from_pretrained(model_args.rank_model_path)
    
    config.max_candi_cnt = model_args.max_candi_cnt # model_args.retriever_n_context * model_args.max_subq_cnt
    config.pad_token_id = rank_tokenizer.pad_token_id
    config.eos_token_id = rank_tokenizer.eos_token_id
    config.topk = model_args.topk
    config.tmp = model_args.tmp
    
    rank_fid_model = GenRanker(config)
    model = AutoModel.from_pretrained(model_args.rank_model_path)
   
    load_names = [name for name , param in model.named_parameters()]
    fid_names = [name for name , param in rank_fid_model.named_parameters()]

    loguru.logger.info(f'For load FID, missing key from T5 is {set(fid_names) - set(load_names)}')
    rank_fid_model.load_state_dict(model.state_dict(), strict=False)
    loguru.logger.info(f'Load params for rank_fid_model from {model_args.rank_model_path}')

    
    model_args.origin_vocab_size = rank_fid_model.config.vocab_size
    loguru.logger.info(f'Ranker: origin_vocab_size = {model_args.origin_vocab_size}')
    for i in range(config.max_candi_cnt):
        rank_tokenizer.add_tokens(model_args.special_token_format.format(i))
        
    rank_tokenizer.add_tokens(model_args.special_query_token)
    rank_tokenizer.add_tokens(model_args.special_sub_token)
    rank_tokenizer.add_tokens(model_args.special_split_token)
    
    addition_add = 3
    
    rank_fid_model.resize_token_embeddings(rank_fid_model.config.vocab_size + config.max_candi_cnt + addition_add)
    rank_fid_model.lm_head = nn.Linear(rank_fid_model.config.d_model, rank_fid_model.config.max_candi_cnt, bias=False)
    loguru.logger.info(f'1 Ranker: new_vocab_size = {rank_fid_model.config.vocab_size}')
    loguru.logger.info('After rank_fid_model resize_token_embeddings lm_head size = {}'.format(rank_fid_model.lm_head.weight.size()))
    D0_id = rank_tokenizer.encode('[D0]')
    Dmax_id = rank_tokenizer.encode(f'[D{config.max_candi_cnt-1}]')
    q_id = rank_tokenizer.encode(f'{model_args.special_query_token}')
    subq_id = rank_tokenizer.encode(f'{model_args.special_sub_token}')
    sep_id = rank_tokenizer.encode(f'{model_args.special_split_token}')
    loguru.logger.info(f'D0_id = {D0_id} Dmax_id = {Dmax_id}')
    loguru.logger.info(f'q_id = {q_id}')
    loguru.logger.info(f'subq_id = {subq_id}')
    loguru.logger.info(f'sep_id = {sep_id}')

    # model, tmp, tokenizer, topk, candi_cnt, origin_vocab_size, l2
    model = Ranker(rank_fid_model, model_args.tmp, rank_tokenizer, model_args.topk, model_args.max_candi_cnt, model_args.origin_vocab_size, model_args.l2, config.decoder_start_token_id, bool(model_args.distrub))
    
    if model_args.load_step > 0:
        load_path = os.path.join(os.path.join(training_args.output_dir, f'checkpoint-{model_args.load_step}'), 'model.safetensors')
        state_dict = load_safetensor_state_dict(load_path)
        loguru.logger.info(f'START TO LOAD MODEL STATE FROM {load_path}')

        model.load_state_dict(state_dict)
        loguru.logger.info(f'LOAD MODEL STATE FROM {load_path} Successfully')
    elif model_args.load_path != '':
        load_path = model_args.load_path
        state_dict = load_safetensor_state_dict(load_path)
        model.load_state_dict(state_dict)
        loguru.logger.info(f'LOAD MODEL STATE FROM {load_path}')
    model.model.gradient_checkpointing_enable()

    def load_retrieve_data(retrieve_path):
        query2candi = dict()
        loguru.logger.info(f'Begin to Load Candidate Infos from {retrieve_path}')
        lines = open(retrieve_path).readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            psgs = data['passages']
            query = data['question']
            doc_strs, subqs = get_psgs_subs(psgs)
            query2candi[query] = {
                "subqs": subqs,
                "doc_strs": doc_strs
            }
        loguru.logger.info(f'Load Candidate Infos from {retrieve_path} Over')
        
        return query2candi
    
    if not model_args.test:
        query2candi_train = load_retrieve_data(data_args.train_retrieve_path)
        query2candi_eval = load_retrieve_data(data_args.eval_retrieve_path)
        train_dataset = MyDataset(
            data_args.train_data,
            query2candi_train,
            model_args.topk,
            model_args.max_subq_cnt,
            model_args.max_subq_cnt_per,
            model_args.max_candi_cnt,
            model_args.q_maxlength,
            model_args.d_maxlength,
            rank_tokenizer,
            model_args.special_token_format,
            model_args.special_query_token,
            model_args.special_sub_token,
            model_args.special_split_token,
        )
        eval_dataset = MyDataset(
            data_args.eval_data,
            query2candi_eval,
            model_args.topk,
            model_args.max_subq_cnt,
            model_args.max_subq_cnt_per,
            model_args.max_candi_cnt,
            model_args.q_maxlength,
            model_args.d_maxlength,
            rank_tokenizer,
            model_args.special_token_format,
            model_args.special_query_token,
            model_args.special_sub_token,
            model_args.special_split_token,
            is_test=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if not model_args.test else test_dataset,
            eval_dataset=eval_dataset if not model_args.test else test_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
    else:
        model = cast_to_precision(model, 'bf16').cuda()
        query2candi_test = load_retrieve_data(data_args.test_retrieve_path)
        test_dataset = MyDataset(
            data_args.test_data,
            query2candi_test,
            model_args.topk,
            model_args.max_subq_cnt,
            model_args.max_subq_cnt_per,
            model_args.max_candi_cnt,
            model_args.q_maxlength,
            model_args.d_maxlength,
            rank_tokenizer,
            model_args.special_token_format,
            model_args.special_query_token,
            model_args.special_sub_token,
            model_args.special_split_token,
            is_test=True,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)
        avg_res = evaluate(model_args, data_args, training_args, model, test_dataloader)
        
        loguru.logger.info(f'For test data, avg div = {avg_res}')
        
def evaluate(model_args, data_args, training_args, model, test_dataloader):
    model.eval()
    from time import time
    with torch.no_grad():
        all_metric_res = []
        
        if not model_args.efficient:
            for data in tqdm(test_dataloader):
                for key in data:
                    data[key] = data[key].cuda()

                _, metric_res = model(**data)

                all_metric_res += metric_res.detach().cpu().numpy().tolist()
        else:
            t1 = time()
            for data in tqdm(test_dataloader):
                for key in data:
                    data[key] = data[key].cuda()

                model(**data, efficient=True)

            t2 = time()
            loguru.logger.info('TEST COST: {}s'.format(t2-t1))
        return np.mean(all_metric_res)
    
def show_grad(net, target_names=None):
    for name, param in net.named_parameters():
        if target_names is not None:
            show = False
            for n in target_names:
                if n in name:
                    show = True
                    break
        else:
            show = True
        if show and param.requires_grad and param.grad is not None:
            if name == 'model.shared.weight':
                loguru.logger.info(f'{name}.grad.size = {param.grad.size()}')
                loguru.logger.info(f'{name}.grad.sum(-1)[-300:] = {param.grad.sum(-1)[-300:]}')
                loguru.logger.info(f'{name}.grad.sum(-1)[:300] = {param.grad.sum(-1)[:300]}')
            loguru.logger.info(f"Parameter Name: {name}")
            loguru.logger.info(f"Gradient: {param.grad}")
    
def show_model_names(net):
    for name, param in net.named_parameters():
        loguru.logger.info(f"Parameter Name: {name} size: {param.size()}")

def compute_metrics(pred):
    metric_np = pred.predictions
    
    div = metric_np.mean()
    
    return {
        "div": div
    } 

def load_t5(times):
    model = AutoModel.from_pretrained('../plms/flan-t5-base')
    for name, param in model.named_parameters():
        loguru.logger.info(f'RAW-{times} plms {name} size  = {param.size()}') 

    
if __name__ == '__main__':
    # load_t5(1)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # load_t5(2)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # load_t5(3)
    MAX_CANDI = model_args.max_candi_cnt 
    loguru.logger.info(f'training_args.report_to = {training_args.report_to}')

    model_args.test = bool(model_args.test)
    model_args.efficient = bool(model_args.efficient)
    
    if 'wandb' in training_args.report_to :
        wandb.login()
        # local_rank = int(os.environ["LOCAL_RANK"])
        local_rank = torch.distributed.get_rank()
        loguru.logger.info(f"local_rank = {local_rank}")
        if local_rank == 0:
            run = wandb.init(
            # set the wandb project where this run will be logged
                project=model_args.wandb_project, 
                name=model_args.wandb_name
            )
            loguru.logger.info(f"Running in project: {run.project}")

    if model_args.is_training:
        train_model(model_args, data_args, training_args)
        loguru.logger.info("start test...")