# coding=utf-8

from __future__ import annotations
import os
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
import torch
import torch.nn as nn
from tqdm import *
from torch.utils.data import DataLoader
import json
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from accelerate import Accelerator
from dataclasses import dataclass, field
from typing import Optional

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, set_seed
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer

from src.llm_models import MyLLM
from src.dataset_rag_evaluation import MyDataset
from src.evaluation import f1_score, exact_match_score, rouge_score, normalize_answer, exact_match_strict_score

import torch.multiprocessing as mp
import os
from time import time
from safetensors import safe_open
import loguru

os.environ['WANDB_API_KEY'] = ''

tqdm.pandas()

DATASET2MAX_SUBQ = {
    "wikipassageqa": 8,
    "wikiasp": 7,
}

DATASET2MAXLEN={
    "wikipassageqa": 500,
    "wikiasp": 1000,
}
DATASET2PROMPT_NEW = {
    "wikipassageqa": "Using the given information, please provide a comprehensive response to the user\'s question. Please rely entirely on the given information to answer the question rather than your knowledge. \n**Given Information:**\n{docs}\n**Question:**\n{query}\n**Response:**\n",
    "wikiasp": "Using the given information, please provide a comprehensive response to the user\'s question. Please rely entirely on the given information to answer the question rather than your knowledge. \n**Given Information:**\n{docs}\n**Question:**\n{query}\n**Response:**\n",
}

DATASET2DEC_PROMPT_NEW = {
    "wikipassageqa": "Using the given information, please respond comprehensively to the user\'s question considering its multi-aspects. **Given Information:**\n{docs}\n**Question:**\n{query}\n**Multi-aspects:**\n{decomposes}\n**Response:**\n",
    "wikiasp": "Using the given information, please respond comprehensively to the user\'s question considering its multi-aspects. **Given Information:**\n{docs}\n**Question:**\n{query}\n**Multi-aspects:**\n{decomposes}\n**Response:**\n",
}
    
CLOSE_DATASET2PROMPT_NEW = {
    "wikipassageqa": "Please provide a comprehensive response to the user\'s question considering its multi-aspects. Generate the answer on one line, and line breaks and periods are considered terminators.\n**Question:**\n{query}\n**Response:**\n",
    "wikiasp": "Please provide a comprehensive response to the user\'s question considering its multi-aspects. Generate the answer on one line, and line breaks and periods are considered terminators.\n**Question:**\n{query}\n**Response:**\n",
}
DECODER_START_TOKENI_ID = 0
def get_psgs_subs(psgs):
    doc_list = []
    all_subqs = []
    for sub_p in psgs:
        subqs = sub_p['subqs']
        p = sub_p['doc']
        
        doc_list.append(p)
        all_subqs.append(subqs)

    return doc_list, all_subqs


def load_combretrieve_data(retrieve_path):
    if 'wikiasp' in retrieve_path:
        
        query2candi = []
        loguru.logger.info(f'Begin to Load LIST Combined Candidate Infos from {retrieve_path}')
        lines = open(retrieve_path).readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            psgs = data['passages']
            query = data['question']
            doc_strs, subqs = get_psgs_subs(psgs)
      
            query2candi.append({
                "subqs": subqs,
                "docs": doc_strs
            })
        loguru.logger.info(f'Load LIST Candidate Infos from {retrieve_path} Over')
    else:
        query2candi = {}
        loguru.logger.info(f'Begin to Load DICT Combined Candidate Infos from {retrieve_path}')
        lines = open(retrieve_path).readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            psgs = data['passages']
            query = data['question']
            doc_strs, subqs = get_psgs_subs(psgs)
            query2candi[query] = {
                "subqs": subqs,
                "docs": doc_strs
            }
        loguru.logger.info(f'Load DICT Candidate Infos from {retrieve_path} Over')

    return query2candi


def load_retrieve_data(retrieve_path):
    if 'wikiasp' in retrieve_path:
        query2candi = []
        loguru.logger.info(f'Begin to Load LIST Candidate Infos from {retrieve_path}')
        lines = open(retrieve_path).readlines()
        for i, line in enumerate(tqdm(lines)):
            data = json.loads(line)
            retrieve = data['passages']
            query = data['question']
            dec_retrieve = data['decomposed_psgs']
      
            query2candi.append({
                'retrieve': retrieve,
                'dec_retrieve': dec_retrieve
            })
            
        loguru.logger.info(f'Load LIST Candidate Infos from {retrieve_path} Over')
    else:
        query2candi = {}
        loguru.logger.info(f'Begin to Load DICT Candidate Infos from {retrieve_path}')
        
        lines = open(retrieve_path).readlines()
        for i, line in enumerate(tqdm(lines)):
            data = json.loads(line)
            retrieve = data['passages']
            query = data['question']
            dec_retrieve = data['decomposed_psgs']
            query2candi[query] = {
                'retrieve': retrieve,
                'dec_retrieve': dec_retrieve
            }
            
        loguru.logger.info(f'Load DICT Candidate Infos from {retrieve_path} Over')

    return query2candi
    
def load_policy_model(args, script_args):
    
    global DECODER_START_TOKENI_ID 

    if args.model_path != '':
        if 'safetensors' in args.model_path:
            raw_state_dict = load_safetensor_state_dict(args.model_path)
            
        else:
            raw_state_dict = torch.load(args.model_path)
        state_dict = {k.replace('model.', ''): raw_state_dict[k] for k in raw_state_dict}
        if 'lm_head.weight' in state_dict:
            add_token = state_dict['lm_head.weight'].size(0)
            loguru.logger.info(f"state_dict['lm_head.weight'].size(0) = {state_dict['lm_head.weight'].size(0)} add_token = {add_token}")
        else:
            add_token = args.retriever_n_context
            loguru.logger.info(f"add_token = {add_token}")
        
    else:
        state_dict = None
        add_token = args.retriever_n_context
        loguru.logger.info(f'args.retriever_n_context = {args.retriever_n_context} add_token = {add_token}')
    
    config = AutoConfig.from_pretrained(args.rank_model_path)
    loguru.logger.info(f'args.rank_model_path = {args.rank_model_path} config = {config}')
    config.max_candi_cnt = args.retriever_n_context
    rank_fid_model = GenRanker(config)
    DECODER_START_TOKENI_ID = config.decoder_start_token_id
    loguru.logger.info(f'DECODER_START_TOKENI_ID = {DECODER_START_TOKENI_ID}')
    
    rank_tokenizer = AutoTokenizer.from_pretrained(args.rank_model_path)
    args.origin_vocab_size = rank_fid_model.config.vocab_size
    loguru.logger.info(f'Ranker: origin_vocab_size = {args.origin_vocab_size}')

    for i in range(add_token):
        rank_tokenizer.add_tokens(args.special_token_format.format(i))
    
    rank_tokenizer.add_tokens(args.special_query_token)
    rank_tokenizer.add_tokens(args.special_sub_token)
    rank_tokenizer.add_tokens(args.special_split_token)
    
    addition_add = 3

    rank_fid_model.resize_token_embeddings(rank_fid_model.config.vocab_size + add_token + addition_add)
    rank_fid_model.lm_head = nn.Linear(rank_fid_model.config.d_model, add_token, bias=False)

    loguru.logger.info(f'1 Ranker: new_vocab_size = {rank_fid_model.config.vocab_size}')
    loguru.logger.info(f'After rank_fid_model resize_token_embeddings lm_head size = {rank_fid_model.lm_head.weight.size()}')
    
    if state_dict is not None:
        rank_fid_model.load_state_dict(state_dict, strict=False)
        
        loguru.logger.info(f'LOAD MODEL STATE FROM {args.model_path} For rank_fid_model and critic_fid_model ')
        
        del state_dict
    
    if not script_args.use_peft:
        critic_fid_model = copy.deepcopy(rank_fid_model)
        ref_model = trl_model_class(critic_fid_model, trust_remote_code=False, torch_dtype=torch.bfloat16)
        device_map = None
        peft_config = None
    else:
        peft_config = script_args.peft_config
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
    
        
    def replace_lora(name):
        return '.'.join([ part + '.default' if 'lora' in part else part  for i, part in enumerate(name.split('.'))])
    
    if script_args.use_peft and script_args.peft_path != '':
        model = get_peft_model(rank_fid_model, peft_config)
        
        peft_state_dict_ = load_safetensor_state_dict(script_args.peft_path)
        peft_state_dict = {replace_lora(key): peft_state_dict_[key] for key in peft_state_dict_}
        
        peft_keys = list(peft_state_dict.keys())
        model_keys = list(model.state_dict().keys())
        
        find_keys = []
        for key in peft_keys:
            if key in model_keys:
                find_keys.append(key)
        loguru.logger.info(f'peft_state dict contains {len(peft_keys)} keys find number of {len(find_keys)} in model_keys')
        
        model.load_state_dict(peft_state_dict, strict=False)

    else:
        
        model = trl_model_class(
        rank_fid_model,
        trust_remote_code=script_args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
        torch_dtype=torch.bfloat16
    )
    
    return model, ref_model, rank_tokenizer

def load_safetensor_state_dict(path):
    tensors = {}
    loguru.logger.info(f'LOAD PATH = {path}')
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


@dataclass
class ScriptArguments:
    use_seq2seq: bool = True
    """whether to use seq2seq models"""
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=64, #16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    model_path: str = 'saved_ckpt/fidranker/flan-t5-base-nq-1-3e-5-TRUEEVAL/checkpoint-5800/model.safetensors'
    peft_path: str = ''

    rank_model_path: str = ''
    reader_model_path: str = ''

    test_data: str = ''
    test_comretrieve_data: str = ''
    test_retrieve_data: str = ''
    
    debug_test_path: str = ''
    
    retriever_n_context: int = 270
    q_maxlength: int = 25
    d_maxlength: int = 300
    topk: int = 10
    
    max_subq_cnt: int = 8
    use_decompose: int = 1
    
    document_format: str = '{title} {text}'
    llm_pmt: str = ''
    dataset: str = 'wikipassageqa'
    psg_pmt: str = '[Source {id}]: [Title: {title}. Content: {text}]\n'
    close_pmt: str = ''
    decom_pmt: str = ''
    
    special_token_format: str = field(default="[D{}]")
    special_query_token: str = field(default="[Q]")
    special_sub_token: str = field(default="[SUBQ]")
    special_split_token: str = field(default="[SEP]")
    
    infer_batch: int = 64
    tmp: float = 0.1
    sample_times:int = 1
    eval_batch: int = 64
    
    use_compsg: int = 1
    
    gen_temperature: float = 0.0
    top_p: float = 0.95
    stop_words: str = '.\t\n'
   
    gpt_model: str = field(default="gpt-3.5-turbo-1106")
    openai_api: str = field(default="")
    apikey: str = field(default="")
    
args = tyro.cli(ScriptArguments)
args.use_decompose = bool(args.use_decompose)
args.use_compsg = bool(args.use_compsg)

set_seed(42)
from src.gen_ranker import GenRanker
from src.diversity_metric_raw import Div_metric_ans_weight, Div_metric_multipredict
    
my_args = args

task = my_args.dataset.split('_')[0]
my_args.llm_pmt = DATASET2PROMPT_NEW[task]
my_args.close_pmt = CLOSE_DATASET2PROMPT_NEW[task]
if my_args.use_decompose:
    my_args.decom_pmt = DATASET2DEC_PROMPT_NEW[task]
    loguru.logger.info(f'my_args.decom_pmt = {my_args.decom_pmt}')
else:
    my_args.decom_pmt = None
my_args.generation_max_length = DATASET2MAXLEN[task]
my_args.max_subq_cnt = DATASET2MAX_SUBQ[task]
if task == 'wikipassageqa' or task == 'eli5':
    my_args.stop_words = None 
    
loguru.logger.info(f'TASK = {task}, max_subq_cnt = {my_args.max_subq_cnt} my_args.generation_max_length = {my_args.generation_max_length}')
loguru.logger.info(f'debug_test_path = {my_args.debug_test_path}')

loguru.logger.info('--------------------ARGUMENTS--------------------')
loguru.logger.info(my_args)

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

model, ref_model, rank_tokenizer = load_policy_model(my_args, args)
model.cuda()

if ref_model is not None:
    ref_model.eval()
    ref_model.cuda()

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
comb_query2candi_test = load_combretrieve_data(my_args.test_comretrieve_data)
query2candi_test = load_retrieve_data(my_args.test_retrieve_data)

num_gpus = torch.cuda.device_count()
loguru.logger.info(f'num_gpus = {num_gpus}')

llm_model = LLM(
    model=args.reader_model_path,
    tensor_parallel_size=num_gpus,
    gpu_memory_utilization=0.5
)
if args.stop_words is not None:
    items = args.stop_words.split("\t")
    loguru.logger.info(f'args.stop_words.split(\\t) = {items}')
else:
    loguru.logger.info(f'args.stop_words = {args.stop_words}')
sampling_params = SamplingParams(
    temperature=args.gen_temperature, 
    max_tokens=args.generation_max_length)
llm_tokenizer = AutoTokenizer.from_pretrained(args.reader_model_path)
    
llm = MyLLM(
    my_args,
    my_args.psg_pmt, 
    my_args.llm_pmt,
    my_args.close_pmt,
    llm_model,
    llm_tokenizer, 
    batch_size=my_args.infer_batch,
    max_seq_len=DATASET2MAXLEN[my_args.dataset.split('_')[0]],
    sample_params=sampling_params
)

generation_kwargs = {
    "do_sample": False,
    "max_new_tokens": my_args.topk,
    "tmp": my_args.tmp
}

def process_response(response, eos_token_id):
    bsz = response.size(0)
    
    ones = torch.ones(bsz, 1)#.to(response.device)
    decode_eos_input_ids = (ones * eos_token_id).long()#.to(device)
    decoder_input_ids = torch.cat([decode_eos_input_ids, response], dim=1)
    
    return decoder_input_ids

def get_doc_str(p):
    title = p['title']
    content = p['text']
    if "section" in p and len(p["section"]) > 0:
        title = f"{p['title']}: {p['section']}"
    doc_str = f'{title} {content}'
    return doc_str

def clean_psgs(psgs):
    new_psgs = []
    for psg in psgs:
        if psg['title'] == '' and psg['text'] == '':
            break
        new_psgs.append(psg)
    return new_psgs
 
def show_res(cnt, scores, div_scores, rankdiv, name):
    loguru.logger.info(f'CNT = {cnt} {name} INFO: AVG F1 = {scores[0] / cnt} | AVG EM = {scores[1] / cnt} | AVG EM_STRICT = {scores[2] / cnt} | AVG ROUGE-1 = {scores[3] / cnt} | AVG ROUGE-2 = {scores[4] / cnt} | AVG ROUGE-L = {scores[5] / cnt} | RANK DIV = {rankdiv / cnt}')
    loguru.logger.info(f'CNT = {cnt} {name} DIV INFO: AVG ROUGE-RECALL-1 = {div_scores[0] / cnt} | AVG ROUGE-RECALL-2 = {div_scores[1] / cnt} | AVG ROUGE-RECALL-L = {div_scores[2] / cnt}')
   
def show_all_res(cnt, all_scores, all_div_scores, all_rankdiv, all_name):
    for i, name in enumerate(all_name):
        show_res(cnt, all_scores[i], all_div_scores[i], all_rankdiv[i], name)

def Test(llm, test_dataloader, generation_kwargs):
    device = torch.device('cuda:0')
    scores_enhance = 0
    div_scores_enhance = 0
    rank_div_enhance = 0

    cnt = 0
    dataloader = test_dataloader

    if args.debug_test_path != '':
        wf = open(args.debug_test_path, 'w') 
        
    model.eval()
    for epoch, batch in enumerate(dataloader):
        time1 = time()
        loguru.logger.info(f"----Test step: {epoch}----")
            
        query_tensors = batch["input_ids"]
        candi_tensors = batch["candi_cnts"]
        
        response_tensors = []
        with torch.no_grad():
            query_tensors = [torch.LongTensor(query_tensor).to(device,dtype=torch.long) for query_tensor in query_tensors]
            candi_tensors = [torch.LongTensor(candi_tensor).to(device,dtype=torch.long).squeeze(-1) for candi_tensor in candi_tensors]
            generation_kwargs['candi_cnts'] = candi_tensors
            
            
            per_bsz = 2
            response_tensors = []
            for i in range(0, len(query_tensors), per_bsz):
                bg = i
                ed = min(i + per_bsz, len(query_tensors))
                generation_kwargs['candi_cnts'] = torch.stack(candi_tensors[bg:ed], dim=0)
                generation_kwargs['input_ids'] = torch.stack(query_tensors[bg:ed], dim=0)
                per_response_tensors = model.generate(**generation_kwargs)
                response_tensors.append(per_response_tensors)
            response_tensors = torch.cat(response_tensors, dim=0).detach().cpu()
        response_tensors_np = response_tensors.numpy() #[batch, topk]

        query_strs, sub_query_strs, batch_comb_passages, answers, sub_answers, all_comb_subqs, all_final_scores = batch['query'], batch['sub_querys'], batch['all_final_psgs'], batch['answers'], batch['sub_answers'], batch['all_comb_subqs'], batch['all_final_scores']
        
        all_list_scores, ideal_scores = Div_metric_multipredict([response_tensors_np], np.array(all_final_scores))[:]
        enh_div_scores = all_list_scores[0] #[:]
        
        batch["response_tensors"] = response_tensors_np
        batch["question"] = query_strs
        
        all_sel_psgs = llm.get_select(batch_comb_passages, response_tensors_np)
        enhance_input_strs = llm.get_llm_input(list(query_strs), sel_passages=all_sel_psgs, decomposed_querys=list(sub_query_strs) if args.use_decompose else None)
        enhance_preds = llm.get_llm_output(enhance_input_strs)

        for i in range(len(enhance_preds)):
            #div rank
            ef1 = f1_score(enhance_preds[i], answers[i], normalize_answer)
            eem = exact_match_score(enhance_preds[i], answers[i], normalize_answer)
            eems = exact_match_strict_score(enhance_preds[i], answers[i], normalize_answer)
            erouge = rouge_score(enhance_preds[i], answers[i])
        
            edivrouge = Div_metric_ans_weight(enhance_preds[i], sub_answers[i])
            
            scores_enhance += np.array([ef1, eem, eems] + list(erouge))
            div_scores_enhance += np.array(edivrouge)
            rank_div_enhance += np.array([enh_div_scores[i]])

            cnt += 1
             
            save_data = {
                "enhance_preds": enhance_preds[i],
                "response_tensors": ','.join([str(did) for did in response_tensors[i]]),
                "question": query_strs[i],
                "sub_querys": sub_query_strs[i],
                "answers": answers[i],
                "sub_answers": sub_answers[i]
            }
            
            wf.write(json.dumps(save_data) + '\n')
                    
        torch.cuda.empty_cache()
        time2 = time()
        loguru.logger.info(f'PER STEP COST {time2-time1}s')
            
    all_scores = [scores_enhance]
    all_div_scores = [div_scores_enhance]
    
    all_rankdiv2 = [rank_div_enhance]
    all_name = ['DIVRANK']
    
    show_all_res(cnt, all_scores, all_div_scores, all_rankdiv2, all_name)
    
    wf.close()
    
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    test_dataset = MyDataset(
        my_args,
        my_args.test_data,
        rank_tokenizer,
        comb_query2candi_test,
        query2candi_test,
        istest=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch, collate_fn=collator, shuffle=False, num_workers=0)

    Test(llm, test_dataloader, generation_kwargs)
