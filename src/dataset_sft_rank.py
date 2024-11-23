from torch.utils.data import Dataset
import pickle
import linecache
# from rank_bm25 import BM25Okapi
import json
import numpy as np
from tqdm import tqdm
from rouge import Rouge
import numpy as np
import copy
from src.diversity_metric_raw import find_doc_scores, get_next_doc_score, get_ideal_score

rouger = Rouge()


class MyDataset(Dataset):
    def __init__(self, top_path, retrieve_path, topk, max_subq_cnt, max_subq_cnt_per, max_candi_cnt, max_q_len, max_d_len, tokenizer, special_token_format, special_query_token, special_sub_token, special_split_token, is_test=False):

        self.top_path = top_path
        
        # self.retrieve_path = retrieve_path
        if isinstance(retrieve_path, str):
            print('We load Retrieve data in Dataset')
            self.query2candi = dict()
            print('Begin to Load Candidate Infos')
            # lines = sum([open(p).readlines() for p in retrieve_path], [])
            lines = open(retrieve_path).readlines()
            for line in tqdm(lines):
                data = json.loads(line)
                psgs = data['passages']
                query = data['question']
                # self.query2candi[query] = get_psgs(psgs) #[p['contents'] for p in psgs]
                doc_strs, subqs = self.get_psgs_subs(psgs)
                self.query2candi[query] = {
                    "subqs": subqs,
                    "doc_strs": doc_strs
                }
        else:
            self.query2candi = retrieve_path
        
        print('Load Candidate Infos Over')
            
        # self.top_res = pickle.load(open(top_path, 'rb'))
        # self.qrel = pickle.load(open(qrel_path, 'rb'))
        print('In dataset Load information OVER')
        self.topk = topk
        self.max_subq_cnt = max_subq_cnt
        self.max_subq_cnt_per = max_subq_cnt_per
        self.max_candi_cnt = max_candi_cnt
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        
        self.max_input_len = min(512, (self.max_q_len+1) * (self.max_subq_cnt_per + 1) + self.max_d_len+1)
        self.special_token_format = special_token_format
        self.special_query_token = special_query_token
        self.special_sub_token = special_sub_token
        self.special_split_token = special_split_token
        
        self.tokenizer = tokenizer
        
        self.is_test = is_test

        self.special_token_ids = [self.tokenizer.convert_tokens_to_ids(self.special_token_format.format(i)) for i in range(max_candi_cnt)]
        self.special_query_id = self.tokenizer.convert_tokens_to_ids(self.special_query_token)
        self.special_sub_id = self.tokenizer.convert_tokens_to_ids(self.special_sub_token)
        self.special_split_id = self.tokenizer.convert_tokens_to_ids(self.special_split_token)
        
        print(f'self.special_token_ids[0] = {self.special_token_ids[0]} \n self.special_token_ids[-1] = {self.special_token_ids[-1]}')
        print(f'self.special_query_id = ', self.special_query_id)
        print(f'self.special_sub_id = ', self.special_sub_id)
        print(f'self.special_split_id = ', self.special_split_id)
        
        if 'train' in top_path:
            self.total_len = len(open(self.top_path).readlines())
        else:
            dev_len = len(open(self.top_path).readlines())
            print('Dev data cnt = ', dev_len)
            self.total_len = dev_len #// 5# min(1000, dev_len)
    
    def get_psgs_subs(self, psgs):
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
    
    def __len__(self):
        return self.total_len
    
    def process_subq_per(self, question, subq):
        # print(f'RANK{i}')
        # print(f'Question = {question}')
        # print(f'subq = {subq}')
        if subq[:len(question)] == question:
            new_subq = copy.deepcopy(subq)[len(question):]

            if new_subq[0] == ':':
                new_subq = new_subq[1:]
        else:
            new_subq = subq
        return new_subq.strip()
    
    def process_subqs(self, question, subqs):
        # print(f'question = {question}\nsubqs = {subqs}')
        if isinstance(subqs[0], list):
            new_subqs = []
            for sub_list in subqs:
                new_sub_list = [self.process_subq_per(question, sub) for sub in sub_list]
            new_subqs.append(new_sub_list)
            
        else:
            new_subqs = [self.process_subq_per(question, sub) for sub in subqs]
        
        return new_subqs
    '''
    def get_random_list(self, all_scores, all_retrieve):
        rank_list = []
        left_scores = []
        cand_idxs = np.arange(len(all_retrieve))
        
        for i in range(1, self.topk):
            #base cnt = i
            base_list = np.random.choice(cand_idxs, i, replace=False).tolist()
            next_scores = get_next_doc_score(base_list, all_scores, self.max_candi_cnt)
            base_list += [-1] * (self.topk-1-i)
            rank_list.append(base_list)
            left_scores.append(next_scores)
        
        return rank_list, left_scores
    '''
    def get_random_list(self, all_scores, all_retrieve):
        # rank_list = []
        # left_scores = []
        cand_idxs = np.arange(len(all_retrieve))
        
        lenth = np.random.choice(np.arange(self.topk-1) + 1)
        base_list = np.random.choice(cand_idxs, lenth, replace=False).tolist()
        next_scores = get_next_doc_score(base_list, all_scores, self.max_candi_cnt)
        base_list += [-1] * (self.topk-1-lenth)
        
        
        return base_list, next_scores
    
    def __getitem__(self, idx):
        # key = self.keys[idx]
        # qid, query = key.split('\t')
        # top_res = self.top_res[key]
        line = linecache.getline(self.top_path, idx+1)
        data = json.loads(line)
        
        question = data['question']
        candi_infos = self.query2candi[question]
        all_rel_subqs = candi_infos['subqs'] # self.process_subqs(question, candi_infos['subqs']) # cut cated original query
        all_retrieve = candi_infos['doc_strs'][:self.max_candi_cnt]
        subqs = data['subqs'] #self.process_subqs(question, data['subqs']) # cut cated original query
        subas = data['subas']
        answer = data['answer']
        
        # print(f'all_rel_subqs = {all_rel_subqs}')
        # print(f'subqs = {subqs}')
        
        # raw_ideal_rank_list = [int(item) for item in data['ideal_rank_list'].split(',')]
        # print(f'ideal_rank_list = {ideal_rank_list}')
        # random_rank_list = data['random_rank_list'] #dynamic build random list
        
        all_scores = find_doc_scores(subas, all_retrieve)
        # print(f'all_scores shape = {np.array(all_scores).shape}')
        # ideal_rank_list, ideal_score = get_ideal_score_metric(all_scores, self.topk)
        ideal_rank_list, ideal_score = get_ideal_score(all_scores, self.topk)
        # print(f'ideal_rank_list = {ideal_rank_list} ideal_score = {ideal_score}')
        # if ideal_rank_list != raw_ideal_rank_list:
        #     print(f'raw_ideal_rank_list = {raw_ideal_rank_list} ideal_rank_list = {ideal_rank_list} ideal_score = {ideal_score} len(subqs) = {len(subqs)} len(set(ideal_rank_list)) = {len(set(ideal_rank_list))} ')
        
        qd_input_ids = []
        for i, psg in enumerate(all_retrieve):
            per_subqs = all_rel_subqs[i]
            
            qd_str = self.special_token_format.format(i)+ ' ' + question + f' {self.special_query_token} ' + f' {self.special_split_token} '.join(per_subqs) + f' {self.special_sub_token} ' + psg
            input_ids = self.tokenizer.encode(qd_str, max_length=self.max_input_len, truncation=True, padding='max_length')
            
            qd_input_ids.append(input_ids)
            
            # if i < 4:
            #     print(f'max input length = {self.max_input_len}')
            #     print(f'per_subqs = {per_subqs}')
            #     print(f'qd_str = {qd_str}')
            #     print(f'input_ids = {input_ids}')
        if len(qd_input_ids) < self.max_candi_cnt:
            pad_cnt = self.max_candi_cnt - len(qd_input_ids)
            pad_input_ids = [0] * self.max_input_len
            qd_input_ids += [pad_input_ids] * pad_cnt
            
        candi_cnts = len(all_retrieve)
        subq_cnts = len(subas)
        
        #process targets:
        #ideal
        #ideal_scores: [topk, max_candi_cnt]
        ideal_scores = []
        for i in range(len(ideal_rank_list)):
            sel_list = ideal_rank_list[:i]
            # print(f'sel_list = {sel_list}, type(sel_list) = {type(sel_list)}')
            next_scores = get_next_doc_score(sel_list, all_scores, self.max_candi_cnt)
            ideal_scores.append(next_scores)
            
            # print(f'Rank-{i}, max next score = {np.max(next_scores)} max next score idx = {np.argmax(next_scores)} ideal rank list[{i}] = {ideal_rank_list[i]}')
        
        if not self.is_test:
            #random:
            #random_rank_list: [topk-1, topk-1]
            #left_scores: [topk-1, max_candi_cnt]
            random_rank_list, left_scores = self.get_random_list(all_scores, all_retrieve) 
            # print('random_rank_list = ',random_rank_list)
    
        qd_input_ids = np.array(qd_input_ids, dtype=np.int64)
        candi_cnts = np.array(candi_cnts, dtype=np.int32)
        # subq_cnts = np.array(subq_cnts, dtype=np.int32)
        ideal_rank_list = np.array(ideal_rank_list, dtype=np.int64)
        ideal_scores = np.array(ideal_scores)
        
        if not self.is_test:
            random_rank_list = np.array(random_rank_list, dtype=np.int64)
            left_scores = np.array(left_scores)
        else:
            random_rank_list = np.array([0])
            all_scores = np.array(all_scores)
            # print('Before pad all_scores, all_scores shape = ', all_scores.shape)
            cur_sub_cnt, cur_doc_cnt = all_scores.shape[:]
            # assert cur_sub_cnt == subq_cnts
            assert cur_doc_cnt == candi_cnts
            
            pad_all_scores = np.ones([self.max_subq_cnt, self.max_candi_cnt]) * -1
            pad_all_scores[:cur_sub_cnt, :cur_doc_cnt] = all_scores
            
            # print('After pad all_scores, pad_all_scores shape = ', pad_all_scores.shape)
            left_scores = pad_all_scores # used to compute metric
        
    
        data = {
            'qd_input_ids': qd_input_ids,
            # 'subq_cnts': 
            'candi_cnts': candi_cnts,
            'labels': ideal_rank_list,
            'ideal_scores': ideal_scores,
            'random_rank_list': random_rank_list,
            'left_scores': left_scores
        }
        
        # for key in data:
        #     print(f'Final Data, {key} shape = {data[key].shape}')

        return data



        