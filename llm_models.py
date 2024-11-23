import torch
from transformers import BertTokenizer, BertModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch.nn.functional as F
import torch.nn as nn
import collections
import re
# import openai
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from multiprocessing import Lock, Pool
import json
import sys
import os

class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code

class OpenAIApiProxy():
    def __init__(self, openai_api, api_key=None):
        self.openai_api = openai_api
        retry_strategy = Retry(
            total=3,  # 最大重试次数（包括首次请求）
            backoff_factor=5, # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504], # 需要重试的状态码列表  
            allowed_methods=["POST"] # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key
    def call(self, params_gpt, headers={}):
        headers['Content-Type'] = headers['Content-Type'] if 'Content-Type' in headers else 'application/json'
        if self.api_key:
            headers['Authorization'] = "Bearer " + self.api_key
        if self.openai_api != 'https://api.openai.com/v1/completions':
            url = self.openai_api + '/v1/chat/completions'
        else:
            url = self.openai_api
        # print('Call url = ', url)
        
        myflag = True
        while(myflag):
            try:
                response = self.session.post(url, headers=headers, data=json.dumps(params_gpt))
                myflag = False
            except Exception as e:
                print("access openai error, sleeping 20s ...")
                print(e)
                sys.stdout.flush()
                time.sleep(20)
        data = json.loads(response.text)
        return data

def get_answer(proxy, model, prompt):

    headers = {}
    prompt_dict = {
        "model": model,
        # "messages": [json.dumps({"role": "system", "content":prompt})]
        # "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content":prompt}],
        "temperature": 0.01
    }

    ans_json = proxy.call(prompt_dict)

    print(ans_json)
    try:
        resp = ans_json["choices"][0]["message"]["content"]
    except:
        if ans_json['error']['code'] == 'context_length_exceeded':
            resp = ''
        else:
            resp = ans_json["choices"][0]["message"]["content"]
    return resp

def gpt_generate_str(proxy, model, input_str):
    answer = get_answer(proxy, model, input_str)

    return answer

def runProcess_gpt(inputs):
    strs, proxy, model = inputs[:]
    res_list = []
    for s in strs:
        res = gpt_generate_str(proxy, model, s)
        res_list.append(res)

    return res_list

def gpt_generate(input_str, pool_size, proxy, model):
    
    input_lists = []
    bsz = len(input_str)
    if pool_size > bsz: pool_size = bsz
    per_cnt = bsz // pool_size 
    

    for i in range(pool_size):
        bg = i * per_cnt
        ed = (i+1) * per_cnt if i < pool_size-1 else bsz
        input_lists.append([input_str[bg:ed], proxy, model])

    pool = Pool(pool_size)
    res_list = pool.map(runProcess_gpt, input_lists)
    final_list = sum(res_list, [])

    return final_list

class MyLLM:
    def __init__(self, opt, psg_pmt, pmt, close_pmt, model, tokenizer, batch_size=8, max_seq_len=32, sample_params=None):
        self.opt = opt
        self.psg_pmt = psg_pmt
        self.prompt = pmt
        self.close_pmt = close_pmt
        self.decom_pmt = opt.decom_pmt
        self.model = model 
        self.reader_tokenizer = tokenizer
        if tokenizer is not None:
            self.reader_tokenizer.pad_token = tokenizer.eos_token
            self.reader_tokenizer.padding_side = "left"
        self.infer_batch = batch_size
        self.max_seq_len = max_seq_len
        self.model_name = "llama"
        self.sample_params = sample_params
        self.pool_size = self.infer_batch

        
    def parse_generate(self, batch_input_ids, batch_generate_ids):
        responses = []

        bsz, max_input = batch_input_ids.size()[:2]
        for i, generated_sequence in enumerate(batch_generate_ids):
            input_ids = batch_input_ids[i]
            text = self.reader_tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    self.reader_tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text.strip())

        return responses
    
    def get_select(self, psgs, actions):
        
        psgs = np.array(psgs)
        actions = np.array(actions).astype(np.int32)
        topk = actions.shape[1]
        bsz, candi_cnt = psgs.shape[:2]
        all_sel_psgs = []
        pre_mask = np.arange(candi_cnt)[None, :].repeat(bsz, axis=0)
        for i in range(topk):
            action = actions[:, i:i+1]
            mask = pre_mask == action # [batch, candi_cnt]
            sel_psgs = psgs[mask] #[batch, str]
            all_sel_psgs.append(sel_psgs)
        all_sel_psgs = np.stack(all_sel_psgs, axis=1)#[batch, topk]
        print('all_sel_psgs size =', all_sel_psgs.shape)
        return all_sel_psgs #
    
    def get_select_diffcnt(self, psgs, actions):
        
        psgs = np.array(psgs)
        actions = np.array(actions).astype(np.int32)
        topk = actions.shape[1]
        bsz, candi_cnt = psgs.shape[:2]
        all_sel_psgs = []
        for i in range(bsz):
            all_sel_psgs.append([])
        pre_mask = np.arange(candi_cnt)[None, :].repeat(bsz, axis=0)
        for i in range(topk):
            action = actions[:, i:i+1]
            for j in range(bsz):
                if action[j] == -1 or action[j] in actions[j,:i]:
                    all_sel_psgs[j].append({'title': '', 'text': ''})
                else:
                    all_sel_psgs[j].append(psgs[j,action[j]][0])
                
        all_sel_psgs = np.array(all_sel_psgs) #[batch, topk]
        return all_sel_psgs #
    
    def get_select_diffcnt_repeat(self, psgs, actions):
        
        psgs = np.array(psgs)
        actions = np.array(actions).astype(np.int32)
        # print('In get_select,psgs shape = {} actions shape = {} '.format(psgs.shape, actions.shape))
        topk = actions.shape[1]
        bsz, candi_cnt = psgs.shape[:2]
        all_sel_psgs = []
        for i in range(bsz):
            all_sel_psgs.append([])
        pre_mask = np.arange(candi_cnt)[None, :].repeat(bsz, axis=0)
        for i in range(topk):
            action = actions[:, i:i+1]
            # mask = pre_mask == action # [batch, candi_cnt]
            # sel_psgs = psgs[mask] #[batch, str]
            # all_sel_psgs.append(sel_psgs)
            for j in range(bsz):
                if action[j] == -1:
                    # continue
                    all_sel_psgs[j].append({'title': '', 'text': ''})
                else:
                    # print(f"psgs[j,action[j]][0] = {psgs[j,action[j]][0]}")
                    all_sel_psgs[j].append(psgs[j,action[j]][0])
                
        all_sel_psgs = np.array(all_sel_psgs) #[batch, topk]
        
        return all_sel_psgs #
    
    def rag_input(self, query, backgrd_psg, decoms=None):
        if decoms is None:
            reader_input_str = self.prompt.format_map({
                'query': query,
                'docs': backgrd_psg
            })
        else:
            reader_input_str = self.decom_pmt.format_map({
                'query': query,
                'docs': backgrd_psg,
                'decomposes': decoms,
                # 'demons': self.demons
            })
        return reader_input_str

    def get_llm_input(self, querys, sel_passages=None, each_step=False, decomposed_querys=None):
        reader_input_strs = [] #each query related to topk input_strs if we set each_step=True
        if sel_passages is not None:
            for i, psgs in enumerate(sel_passages):
                psg_cnt = 1
                backgrd_psg = ''
                
                if decomposed_querys is not None:
                    per_decoms = decomposed_querys[i]
                    decom_str = '\n'.join(per_decoms)
                else:
                    decom_str = None
                    
                for psg in psgs:
                    if psg is None or psg == '' or (psg['title'] == '' and psg['text'] == ''): 
                        continue
                    pmt_psg = self.psg_pmt.format_map({
                        'id': psg_cnt,
                        'title': psg['title'],
                        'text': psg['text']
                        })
                    backgrd_psg += pmt_psg
                    if each_step:
                        inner_reader_input_str = self.rag_input(querys[i], backgrd_psg, decom_str)
                        
                        reader_input_strs.append(inner_reader_input_str)
                        
                    psg_cnt += 1
                if not each_step:
                    reader_input_str = self.rag_input(querys[i], backgrd_psg, decom_str)
                    reader_input_strs.append(reader_input_str)
        else:
            for q in querys:
                reader_input_str = self.close_pmt.format_map({
                    'query': q,
                    'decomposes':decomposed_querys
                })
                reader_input_strs.append(reader_input_str)
        return reader_input_strs
    
    @torch.no_grad()
    def get_llm_output(self, reader_input_strs):
        bsz = len(reader_input_strs)
        
        out_txt_seqs = []
        for i in range(0, bsz, self.infer_batch):
            bg, ed = i, i+self.infer_batch
            reader_output = self.model.generate(
                reader_input_strs[bg: ed], self.sample_params
            )
            per_out_txt_seqs = [output.outputs[0].text for output in reader_output]
            out_txt_seqs += per_out_txt_seqs
        
        return out_txt_seqs