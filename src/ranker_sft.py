
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
# import gym
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F 
from transformers.modeling_outputs import SequenceClassifierOutput
from src.diversity_metric_raw import Div_metric

class Ranker(nn.Module):
    def __init__(self, model, tmp, tokenizer, topk, candi_cnt, origin_vocab_size, l2, decoder_start_token_id, distrub=False):
        super(Ranker, self).__init__()
        self.model = model #encoder decoder
        
        self.tmp = tmp
        self.tokenizer = tokenizer

        self.topk = topk
        self.candi_cnt = candi_cnt
        self.origin_vocab_size = origin_vocab_size

        self.doc_special_tokens = [self.tokenizer.encode('[D{}]'.format(i))[1] for i in range(candi_cnt)]
        self.eos_token_id = 0 # 0 is T5's decoder_start_token_id self.tokenizer.eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.distrub = distrub
        print('In Rankmodel, distrub = ', self.distrub)
        print(f'self.decoder_start_token_id = {decoder_start_token_id}')
        
        self.l2 = l2 # the weight of disturb loss
        self.generation_kwargs = {
            'max_new_tokens': self.topk,
            'tmp': self.tmp,
            'do_sample': False,
        }
        print('self.generation_kwargs = ', self.generation_kwargs)
        
    def mask_selected_docs(self, logits, sel_tokens, value=-torch.inf):
        logits = logits.scatter_(1, sel_tokens, value)
        return logits
    
    def decode_qdid(self, sel_idx):
        #sel_index:[batch,]
        sel_qid = sel_idx // self.candi_cnt
        sel_did = sel_idx % self.candi_cnt
        
        return sel_qid, sel_did
        
    def get_encode_atten_mask_given_target(self, doc_mask, generated_tokens):
        # doc_mask: [batch, candi_cnt]
        # generated_tokens: [batch, topk]
        
        bsz, topk = generated_tokens.size()
        doc_mask_3d = doc_mask.unsqueeze(1).repeat([1, topk, 1])
        
        
        for i in range(topk-1):
            doc_mask_3d[:, i+1, :] = self.mask_selected_docs(doc_mask_3d[:, i+1, :], generated_tokens[:, i:i+1].long(), 0)
        
        return doc_mask_3d
    
    def forward(
        self, 
        qd_input_ids,
        candi_cnts,
        labels,
        ideal_scores,
        random_rank_list=None,
        left_scores=None,
        efficient=False,
    ):
        # qd_input_ids: [batch, max_candi_cnt, max_input_len]:[D{i}]Query[Q]sub1[SEP]...subl[SUBQ]document 
        # candi_cnts: [batch, 1] indicate the candi_cnt of each query.
        # labels: [batch, topk]
        # ideal_scores: [batch, topk, max_candi_cnt]
        # random_rank_list: [batch, topk-1, topk-1] #[previous count from 1 to topk-1]
        # left_scores: [batch, topk-1, max_candi_cnt]
        
        device = qd_input_ids.device
        if self.training == False:
            
            self.generation_kwargs['candi_cnts'] = candi_cnts
            self.generation_kwargs['do_sample'] = False
            
            generated_tokens = self.model.generate(
                input_ids=qd_input_ids,
                **self.generation_kwargs
            )
            if not efficient:
                prediction = generated_tokens.detach().cpu().numpy()
                all_scores = left_scores.detach().cpu().numpy()

                metric_res = Div_metric(prediction, all_scores)
            
                return [torch.FloatTensor([0]).to(qd_input_ids.device), torch.FloatTensor(metric_res).to(qd_input_ids.device)] #[batch]
            else:
                return None
        else:
            # forward
            outputs = self.model(
                input_ids=qd_input_ids, 
                candi_cnts=candi_cnts,
                labels=labels,
                left_doc_probs=ideal_scores,
                ideal=True,
                pre_list_cnt=None,
                return_dict=True,
                use_cache=False,
            )
            
            past_key_values = outputs.past_key_values
            decoder_hidden_states = outputs.decoder_hidden_states
            decoder_attentions = outputs.decoder_attentions
            cross_attentions = outputs.cross_attentions
            encoder_last_hidden_state = outputs.encoder_last_hidden_state #[batch, candi_cnt, dim]
            encoder_hidden_states = outputs.encoder_hidden_states
            encoder_attentions = outputs.encoder_attentions
            
            logits = outputs.logits
            ideal_loss = outputs.loss
            if self.distrub:
                
                batch, max_base_list_cnt = random_rank_list.size()
                max_candi_cnt = left_scores.size(-1)

                encoder_outputs = [
                    encoder_last_hidden_state, 
                ] # [batch, max_candi_cnt, dim]

                # labels: [batch, topk]
                # ideal_scores: [batch, topk, max_candi_cnt]
                # random_rank_list: [batch, topk-1, topk-1] #[previous count from 1 to topk-1]
                # left_scores: [batch, topk-1, max_candi_cnt]
                pre_list_cnt = (random_rank_list >= 0).sum(-1) #[batch]
                
                outputs = self.model(
                    input_ids=qd_input_ids, 
                    candi_cnts=candi_cnts,
                    labels=random_rank_list,
                    encoder_outputs=encoder_outputs, ## add encoder_outputs
                    left_doc_probs=left_scores,
                    ideal=False,
                    pre_list_cnt=pre_list_cnt,
                    return_dict=True,
                )

                disturb_loss = outputs.loss

                loss = ideal_loss + self.l2 * disturb_loss
                
            else:
                loss = ideal_loss
            return SequenceClassifierOutput(
                loss=loss,
            )
