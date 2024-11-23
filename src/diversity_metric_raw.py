
import json
import numpy as np
from tqdm import tqdm
from rouge import Rouge
import copy
from src.evaluation import f1_score, exact_match_score, rouge_score, normalize_answer, exact_match_strict_score

rouger = Rouge()

def get_rouge(prediction, answer):
    score = rouger.get_scores(prediction, answer)[0]
    r1 = score['rouge-1']['r']
    r2 = score['rouge-2']['r']
    rl = score['rouge-l']['r']

    return [r1, r2, rl]

def find_doc_metric(answer, passages):
    all_scores = []
    for i, psg in enumerate(passages):
        score = np.max(get_rouge(psg, answer))

        all_scores.append(score)

    return all_scores

def find_doc_scores(subas, all_retrieve):

    all_scores = []
    for i, sa in enumerate(subas):
        per_scores = find_doc_metric(sa, all_retrieve)
        all_scores.append(per_scores)

    return all_scores

def get_next_doc_score(rank_list, all_scores, max_candi_cnt=None, reweight=True):
    # Next scores are only used to provide relative scores, so we don't need to norm them. Moreover, the normed scores will make diversity score too low.
    all_scores_np = np.array(all_scores)
    left_scores = copy.deepcopy(all_scores_np)
    sub_cnt = all_scores_np.shape[0]
    try:
        max_sub_cover = np.max(all_scores, axis=1)
    except:
        print(f'In get_next_doc_score all_scores shape = {np.array(all_scores).shape}')
        max_sub_cover = np.max(all_scores, axis=1)
    if len(rank_list) > 0:
      
        sub_cover = np.zeros(all_scores_np.shape[0])
        for j in range(len(rank_list)):
            sub_cover = np.max(np.stack([sub_cover, all_scores_np[:, rank_list[j]]], axis=1), axis=1)
            if (sub_cover == max_sub_cover).sum() == all_scores_np.shape[0]:
                sub_cover = np.zeros(all_scores_np.shape[0])
        sub_cover_norm = sub_cover #/ max_sub_cover

    else:
        sub_cover = np.zeros(sub_cnt)
        sub_cover_norm = np.zeros(sub_cnt)
    sub_uncover = 1 - sub_cover_norm
    if sub_uncover.sum() > 0:
        sub_uncover_norm = sub_uncover / sub_uncover.sum() 
    else:
        sub_uncover_norm = sub_uncover 
        
    if sub_uncover_norm.sum() == 0 and reweight:
        
        sub_uncover_norm = sub_uncover_norm + 1 / all_scores_np.shape[0] # all one
    weight_left_score = ((sub_uncover_norm[:, None]) * left_scores).sum(0)#[candi_cnt]

    if max_candi_cnt is not None and len(weight_left_score) < max_candi_cnt:
        pad_weight_left_score = np.concatenate([weight_left_score, np.zeros(max_candi_cnt - len(weight_left_score))], axis=0)

    else:
        pad_weight_left_score = weight_left_score
    return pad_weight_left_score

def get_ideal_score(all_scores, k, reweight=True):
    rank_list = []
    # left_scores = []
    sum_score = 0.0
    max_scores = []
    all_scores = np.array(all_scores)
    for i in range(k):
        next_scores = get_next_doc_score(rank_list, all_scores, reweight=reweight)
        if len(rank_list) > 0:
            next_scores = np.array(next_scores)
            next_scores[np.array(rank_list,dtype=np.int32)] = -10
        sel_idx = np.argmax(next_scores)
        max_score = np.max(next_scores)
        # assert sel_idx not in rank_list # which means all scores of left documents are zero
        
        max_scores.append(all_scores[:, sel_idx])
            
        sum_score += max_score
        rank_list.append(sel_idx)
    
    final_score = sum_score
    return rank_list, final_score
    
def Div_metric(predictions, all_scores):
    # predictions: [batch, topk]
    # all_scores: [batch, max_subq_cnt, max_candi_cnt]
    batch, topk = predictions.shape[:]
    max_subq_cnt, max_candi_cnt = all_scores.shape[1:]
    all_metric_scores = []
    for i in range(batch):
        cur_predictions = predictions[i]
        cur_all_scores = all_scores[i]
        
        valid_subq = np.where(cur_all_scores.sum(1) > -1 * max_candi_cnt, 1, 0).sum()
        valid_doc = np.where(cur_all_scores.sum(0) > -1 * max_subq_cnt, 1, 0).sum()
    
        valid_cur_all_scores = cur_all_scores[:valid_subq, :valid_doc] #[true_subcnt, true_candi_cnt]
        
        ideal_list, ideal_score = get_ideal_score(valid_cur_all_scores, topk)
        
        sum_score = 0.0
        for t in range(topk):
            next_scores = get_next_doc_score(cur_predictions[:t], valid_cur_all_scores, reweight=False)
            sel_idx = cur_predictions[t]
            sum_score += next_scores[sel_idx]
        final_score = sum_score
        norm_score = final_score / ideal_score
        
        all_metric_scores.append(norm_score)
    
    return np.array(all_metric_scores)

def Div_metric_multipredict(predictions, all_scores):
    # predictions: [batch, topk]
    # all_scores: [batch, max_subq_cnt, max_candi_cnt]
    batch, topk = predictions[0].shape[:]
    max_subq_cnt, max_candi_cnt = all_scores.shape[1:]
    
    all_metric_scores = []
    all_ideal_score = []
    for i in range(len(predictions)):
        all_metric_scores.append([])
        
    for i in range(batch):
        
        cur_all_scores = all_scores[i]
        
        valid_subq = np.where(cur_all_scores.sum(1) > -1 * max_candi_cnt, 1, 0).sum()
        valid_doc = np.where(cur_all_scores.sum(0) > -1 * max_subq_cnt, 1, 0).sum()
    
        valid_cur_all_scores = cur_all_scores[:valid_subq, :valid_doc] #[true_subcnt, true_candi_cnt]
        
        ideal_list, ideal_score = get_ideal_score(valid_cur_all_scores, topk)
        
        for j in range(len(predictions)):
            cur_predictions = predictions[j][i]

            sum_score = 0.0
            for t in range(topk):
                next_scores = get_next_doc_score(cur_predictions[:t], valid_cur_all_scores, reweight=False)
                
                sel_idx = cur_predictions[t]
                sum_score += next_scores[sel_idx]
            final_score = sum_score
            norm_score = final_score / ideal_score
            
            all_metric_scores[j].append(norm_score)
        all_ideal_score.append(ideal_score)

    return np.array(all_metric_scores), np.array(all_ideal_score)

def Div_metric_ans(prediction, sub_answers):
    all_rouge = np.array([0.0, 0.0, 0.0])
    sub_cnt = len(sub_answers)
    for suba in sub_answers:
        rouge = np.array(rouge_score(prediction, [suba], metric="r"))
        
        all_rouge += rouge
        
    divrouge = all_rouge / sub_cnt
    
    return divrouge

def Div_metric_ans_weight(prediction, sub_answers):
    all_rouge = np.array([0.0, 0.0, 0.0])
    sub_cnt = len(sub_answers)
    
    weights = [len(normalize_answer(suba).split()) for suba in sub_answers]
    weights = np.array(weights) / np.sum(weights)
    
    for i, suba in enumerate(sub_answers):
        rouge = np.array(rouge_score(prediction, [suba], metric="r"))
        
        all_rouge += weights[i] * rouge
        
    divrouge = all_rouge
    
    return divrouge