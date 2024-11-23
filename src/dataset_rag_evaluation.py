from torch.utils.data import Dataset
import json
import numpy as np
import copy
from src.diversity_metric_raw import find_doc_scores
import loguru
class MyDataset(Dataset):
    def __init__(self, opt, path, tokenizer, comb_query2candi, query2candi, istest=False):
        self.opt = opt
        self.path = path
        self.task = opt.dataset.split('_')[0]
        self.tokenizer = tokenizer
        self.comb_query2candi = comb_query2candi
        self.query2candi = query2candi
  
        if istest:
            self.datas = [json.loads(line) for line in open(self.path).readlines()]
            loguru.logger.info(f'Test Dataset len = {len(self.datas)}')
        else:
            self.datas = [json.loads(line) for line in open(self.path).readlines()]
            
            loguru.logger.info(f'Train OR DEV Dataset len = {len(self.datas)}')
        
        self.topk = opt.topk
        self.max_subq_cnt = opt.max_subq_cnt
        self.max_candi_cnt = opt.retriever_n_context
        self.max_q_len = opt.q_maxlength
        self.max_d_len = opt.d_maxlength
        
        self.max_input_len = min(512, (self.opt.q_maxlength+1) * (self.opt.max_subq_cnt + 1) + self.opt.d_maxlength+1)
        self.special_token_format = opt.special_token_format
        self.special_query_token = opt.special_query_token
        self.special_sub_token = opt.special_sub_token
        self.special_split_token = opt.special_split_token
        
        self.is_test = istest

        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(self.opt.special_token_format.format(i)) for i in range(self.max_candi_cnt)]
        self.special_query_id = self.tokenizer.convert_tokens_to_ids(self.opt.special_query_token)
        self.special_sub_id = self.tokenizer.convert_tokens_to_ids(self.opt.special_sub_token)
        self.special_split_id = self.tokenizer.convert_tokens_to_ids(self.opt.special_split_token)
        
        loguru.logger.info(f'In MyDataset self.special_token_ids[0] = {self.special_token_ids[0]} \n self.special_token_ids[-1] = {self.special_token_ids[-1]}')
        loguru.logger.info(f'In MyDataset self.special_query_id = {self.special_query_id}')
        loguru.logger.info(f'In MyDataset self.special_sub_id = {self.special_sub_id}')
        loguru.logger.info(f'In MyDataset self.special_split_id = {self.special_split_id}')
        
        self.total_len = len(self.datas)
   
    def __len__(self):
        return self.total_len
    
    def get_seprank_scores_repeat(self, idx, all_comb_psgs):
        data = self.datas[idx]
        query = data["question"]
        if isinstance(self.query2candi, list):
            sep_retrieve = self.query2candi[idx]['dec_retrieve']
        else:
            sep_retrieve = self.query2candi[query]['dec_retrieve']
        rel_rank_list = self.seqrel_ranker_list[idx]
        
        subqs = data['subqs']
        subas = data['subas']
        sub_cnt = len(subqs)
    
        assert len(sep_retrieve) == sub_cnt
        
        per_cnt = int(np.around(self.topk / sub_cnt))
        res_cnt = 0
        results = []
        
        for i in range(sub_cnt):
            cur_cnt = per_cnt if i < sub_cnt - 1 else self.topk - res_cnt
            
            per_list = rel_rank_list[i]
            results += list(np.array(sep_retrieve[i])[np.array(per_list[:cur_cnt], dtype=np.int32)])
                
            res_cnt += cur_cnt
            
        rank_list = []
        for res in results:
            rank_list.append(all_comb_psgs.index(res))
        
        return rank_list
    
    def index(self, item, lists):
        try:
            idx = lists.index(item)
        except:
            idx = -1
        return idx
    
    def get_str_data(self, idx):
        example = self.datas[idx]
        
        query = example["question"]
        assert 'answers' in example or 'answer' in example
        if 'answers' in example:
            answers = example['answers']
        else:
            answers = example['answer']
        
        if isinstance(answers[0], dict):
            answers = [a['answer'] for a in answers]
            
        if self.task == 'wikipassageqa':
            answers = '\n'.join(answers)
            answers = [answers]
            
        sub_querys = example['subqs']
        sub_answers = example['subas']
        
        
        if isinstance(self.comb_query2candi, list):
            all_comb_psgs = self.comb_query2candi[idx]['docs'][:self.opt.retriever_n_context]
            all_comb_subqs = self.comb_query2candi[idx]['subqs'][:self.opt.retriever_n_context]
        else:
            all_comb_psgs = self.comb_query2candi[query]['docs'][:self.opt.retriever_n_context]
            all_comb_subqs = self.comb_query2candi[query]['subqs'][:self.opt.retriever_n_context]
   
        all_final_psgs = copy.deepcopy(all_comb_psgs)
        
        all_final_scores = np.array(find_doc_scores(sub_answers, [self.get_doc_str(p) for p in all_final_psgs]))
        max_candi = self.opt.retriever_n_context + 100
        if len(all_final_psgs) < max_candi:
            pad_cnt = max_candi - len(all_final_psgs)
            pad_psg = {'title': '', 'text': ''}
            all_final_psgs += [pad_psg] * pad_cnt
            
        pad_all_scores = np.ones([self.max_subq_cnt, max_candi]) * -1
        pad_all_scores[:all_final_scores.shape[0], :all_final_scores.shape[1]] = all_final_scores
        all_final_scores = pad_all_scores
            
        all_final_psgs = np.array(all_final_psgs)
  
        return query, sub_querys, all_final_psgs, answers, sub_answers, all_comb_subqs, all_final_scores
        
    
    def get_doc_str(self, p):
        title = p['title']
        content = p['text']
        if "section" in p and len(p["section"]) > 0:
            title = f"{p['title']}: {p['section']}"
        doc_str = f'{title} {content}'
        return doc_str
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        question = data['question']
        if isinstance(self.comb_query2candi, list):
            candi_infos = self.comb_query2candi[idx]
        else:
            candi_infos = self.comb_query2candi[question]
        if not self.opt.use_compsg:
            new_subqs = []
            new_docs = []
            if isinstance(self.comb_query2candi, list):
                candis = self.query2candi[idx]['retrieve']
            else:
                candis = self.query2candi[question]['retrieve']
                
            for doc in candis:
                new_docs.append(doc)
                subqs = [question]
                if doc in candi_infos['docs']:
                    subqs = candi_infos['subqs'][candi_infos['docs'].index(doc)]
                new_subqs.append(subqs)
            candi_infos = {
                'subqs' : new_subqs,
                'docs' : new_docs
            }
                    
        all_rel_subqs = candi_infos['subqs']
        all_retrieve = [self.get_doc_str(p) for p in candi_infos['docs']][:self.opt.retriever_n_context]
        subqs = data['subqs'] 
        
        qd_input_ids = []
        for i, psg in enumerate(all_retrieve):
            per_subqs = all_rel_subqs[i]
            
            qd_str = self.special_token_format.format(i)+ ' ' + question + f' {self.special_query_token} ' + f' {self.special_split_token} '.join(per_subqs) + f' {self.special_sub_token} ' + psg
            input_ids = self.tokenizer.encode(qd_str, max_length=self.max_input_len, truncation=True, padding='max_length')
            
            qd_input_ids.append(input_ids)
         
            
        if len(qd_input_ids) < self.max_candi_cnt:
            pad_cnt = self.max_candi_cnt - len(qd_input_ids)
            pad_input_ids = [0] * self.max_input_len
            qd_input_ids += [pad_input_ids] * pad_cnt
            
            
        candi_cnts = len(all_retrieve)

        qd_input_ids = np.array(qd_input_ids, dtype=np.int64)
        candi_cnts = np.array([candi_cnts], dtype=np.int32)
        
        query, sub_querys, all_final_psgs, answers, sub_answers, all_comb_subqs,all_final_scores = self.get_str_data(idx)
    
    
        data = {
            'index': idx,
            'input_ids': qd_input_ids, #[max_candi, max_len]
            'candi_cnts': candi_cnts, #[]
            'query': query,
            'sub_querys': sub_querys, 
            'all_final_psgs': all_final_psgs, 
            'answers': answers, 
            'sub_answers': sub_answers, 
            'all_comb_subqs': all_comb_subqs, 
            'all_final_scores': all_final_scores
        }
        
        return data


        