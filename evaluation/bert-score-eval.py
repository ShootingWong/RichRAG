import sys
sys.path.append('..')
from src.evaluation import f1_score, exact_match_score, rouge_score, normalize_answer, exact_match_strict_score
from src.diversity_metric_raw import Div_metric_ans, Div_metric_ans_weight, find_doc_scores, Div_metric_multipredict
import evaluate
import json

import numpy as np
from tqdm import tqdm
import json
import sys 

def get_bert_score(bertscore, predictions, references):
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return results

def get_scores(bertscore, datas):
    keys = ['enhance_preds']
    results = [0 for k in keys]
    
    all_preds = [[] for k in keys]
    all_ans = []
    for data in tqdm(datas):
        subans = data['sub_answers']
        ans = '\n'.join(subans)
        if ans == '':
            ans = '\n'.join(data['answers'])

        all_ans.append(ans)
        for i, key in enumerate(keys):
            if key in data:
                pred = data[key]
            else:
                pred = ''
            all_preds[i].append(pred)
       
    i = 0
    for key in tqdm(keys):
        res = get_bert_score(bertscore, all_preds[i], all_ans)
        print(f'{key} result: ', '\t'.join([f'{key}: {np.mean(res[key])}' for key in ['precision', 'recall', 'f1']]))
        i += 1
        
def load_data(path):
    return [json.loads(line) for line in open(path).readlines()]

# bertscore = evaluate.load("evaluate-main/metrics/bertscore") #, module_type="metric")  
bertscore = evaluate.load("bertscore")

path = sys.argv[1]
print(f'path: {path}')
datas = [json.loads(line) for line in open(path).readlines()]
get_scores(bertscore, datas)


'''
path_list = [
    '../outputs/wikiasp-golden-llama-2-13b-chat-hf.jsonl'
]
for path in path_list:
    print(f'path: {path}')
    datas = [json.loads(line) for line in open(path).readlines()]
    get_scores(bertscore, datas)
'''

  