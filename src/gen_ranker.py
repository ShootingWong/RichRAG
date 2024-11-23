# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import types

import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils import logging
import numpy as np

# from src.modeling_t5 import T5ForConditionalGeneration, T5Stack
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from time import time

logger = logging.get_logger(__name__)


class GenRankerStack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        generated_tokens=None
    ):
        if input_ids is not None:
            bsz = input_ids.size(0)
        else:
            bsz = inputs_embeds.size(0)
        if not self.is_decoder:
            input_ids = input_ids.view(bsz * self.config.max_candi_cnt, -1)
            if attention_mask is not None:
                attention_mask = attention_mask.view(attention_mask.size(0) * self.config.max_candi_cnt, -1)
            # print('In T5Encoder, input_ids size = {} attention_mask size = {}'.format(input_ids.size(), attention_mask.size()))
        # t1 = time()
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # t2 = time()
        # if self.is_decoder:
            # print(f'Decoder forward cost {t2-t1}')

        if not self.is_decoder:
            if not return_dict:
                last_hidden_states = output[0] # [bsz * candi_cnt, max_qd_length, dim]
                # print('In T5Encoder, last_hidden_states size = {} self.config.max_candi_cnt = {}'.format(last_hidden_states.size(), self.config.max_candi_cnt))

                last_hidden_state = last_hidden_states.view(bsz, self.config.max_candi_cnt, -1, last_hidden_states.size(-1))[:, :, 0, :]# [bsz, candi_cnt, dim]
                # print('In T5Encoder, cut last_hidden_states size = {} '.format(last_hidden_states.size()))
                output = tuple(
                    last_hidden_state,
                    # *output[1:],
                )
            else:
                last_hidden_state = output.last_hidden_state
                output.last_hidden_state = last_hidden_state.view(bsz, self.config.max_candi_cnt, -1, last_hidden_state.size(-1))[:, :, 0, :]# [bsz, candi_cnt, dim]
        
        return output
        

class GenRanker(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        # r"encoder\.embed_tokens\.weight",
        # r"decoder\.embed_tokens\.weight",
        # r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = GenRankerStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = GenRankerStack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def mask_selected_docs(self, logits, sel_tokens, value=-torch.inf):
        logits = logits.scatter_(1, sel_tokens, value)
        return logits
    
    def get_encode_atten_mask_given_target(self, doc_mask, generated_tokens):
        # doc_mask: [batch, candi_cnt]
        # generated_tokens: [batch, topk]
        
        bsz, topk = generated_tokens.size()
        doc_mask_3d = doc_mask.unsqueeze(1).repeat([1, topk+1, 1])
        
        for i in range(topk):
            doc_mask_3d[:, i+1, :] = self.mask_selected_docs(doc_mask_3d[:, i+1, :], generated_tokens[:, :i+1].long(), 0)
            
        return doc_mask_3d
    
    def per_token_loss(self, tmp, per_logits, per_doc_ideal_probs, per_doc_mask):
        # In this version we use the hinge loss optimize the model
        # per_doc_mask, set the previously selected docs and invalid candi_docs(pad for max_candi) as 0
        pad_max_candi_cnt = per_logits.size(-1)
        
        per_logits_1 = per_logits.unsqueeze(1).repeat([1, pad_max_candi_cnt, 1])
        per_logits_2 = per_logits.unsqueeze(-1).repeat([1, 1, pad_max_candi_cnt])
        per_doc_ideal_probs_1 = per_doc_ideal_probs.unsqueeze(1).repeat([1, pad_max_candi_cnt, 1])#[batch, max_candi_cnt]-->[batch, max_candi_cnt, max_candi_cnt]
        pair_mask = per_doc_ideal_probs.unsqueeze(-1) > per_doc_ideal_probs_1 #[batch, max_cadi_cnt, max_candi_cnt] for currend doc, all docs whose score is bigger/equal than/to me is not used to calculate mom
        
        pair_mask[(~per_doc_mask).unsqueeze(-1).repeat([1, 1, pad_max_candi_cnt])] = False # mask the unvalid docs
        
        ranks = ((-per_doc_ideal_probs).argsort(-1)[:, :10]).unsqueeze(-1).repeat([1,1,pad_max_candi_cnt]) #[batch, topk]
        hinge = 1
        
        hinge_loss = torch.relu(per_logits_1[pair_mask] - per_logits_2[pair_mask] + hinge)#.sum() / pair_mask.float().sum()
        
        max_idx = per_doc_ideal_probs.max(-1, keepdim=True)[1] #[batch, 1]
        per_probs = F.softmax(per_logits, dim=-1)
        max_prob = torch.gather(per_probs, 1, max_idx).squeeze(-1)
     
        valid_prob = max_prob > 0
        prob_loss = -torch.log(max_prob[valid_prob])
        
        return hinge_loss, prob_loss

    def forward_post(self, labels, logits, doc_mask, tmp, max_candi_cnt, left_doc_probs, candi_cnts, ideal=True, pre_cnt=None):
        logits = logits / tmp #[:, :-1, :]#[batch, topk, max_candi_cnt]

        logits[:, :max_candi_cnt] = logits[:, :max_candi_cnt].masked_fill_((~doc_mask).bool(), -torch.inf)
        logits[:, max_candi_cnt:] = -torch.inf
        
            
        action_probs = F.softmax(logits, dim=-1) #[batch, topk, max_candi_cnt]
        
        batch = action_probs.size(0)
        pad_max_candi_cnt = action_probs.size(-1)
        
        if ideal:
            topk = labels.size(1)
            hlosses = 0.0
            plosses = 0.0
            hcnt = 0
            pcnt = 0
            # we need to calculate loss for each doc_token in the ideal_ranking list
            
            for i in range(topk):
                per_probs = action_probs[:, i, :] #[batch, topk, max_candi_cnt]-->[batch, max_candi_cnt]
                per_logits = logits[:, i, :] #[batch, topk, max_candi_cnt]-->[batch, max_candi_cnt]
                per_doc_ideal_probs = left_doc_probs[:, i, :] #[batch, max_candi_cnt]
                
                # mask selected docs in per_probs
                sel_tokens = labels[:, :i]
                b = 0
                hinge_loss, prob_loss = self.per_token_loss(tmp, per_logits, per_doc_ideal_probs, doc_mask[:, i, :])

                hlosses += hinge_loss.sum()
                hcnt += (hinge_loss > 0).sum()
                plosses += prob_loss.sum()
                pcnt += prob_loss.size(0)
            
            loss = plosses / pcnt 
            loss = loss + hlosses / hcnt 
            
        else:
            # we only need calculate loss for the last token(which is not appearing in the ranking list
            # action_probs: [batch, topk, max_candi_cnt]
            # left_doc_probs: [batch, max_candi_cnt]
            # labels: previos list, [batch, topk-1]
            # pre_cnt: [batch]
            # left_doc_probs: [batch, max_candi_cnt]
            
            sel_mask = []
    
            for i in range(batch):
                sel_mask.append(doc_mask[i, pre_cnt[i], :])
            sel_mask = torch.stack(sel_mask, dim=0) #[batch, max_candi_cnt]
            
            current_probs = torch.gather(action_probs, 1, pre_cnt.unsqueeze(-1).unsqueeze(-1).repeat([1,1,action_probs.size(-1)])) #[batch, topk, max_candi_cnt]-->[batch, 1, max_candi_cnt]
             
            hinge_loss, prob_loss = self.per_token_loss(tmp, current_probs.squeeze(1), left_doc_probs, sel_mask)
            hloss = hinge_loss.sum() / (hinge_loss > 0).sum()
            ploss = prob_loss.sum() / prob_loss.size(0)
            
            loss = hloss + ploss
        
        return loss

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        critic=False,
        emb=True,
        candi_cnts=None,
        generate=False,
        left_doc_probs=None,
        ideal=True,
        pre_list_cnt=None,
        loss=True,
        # doc_step=0,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> # training
        >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
        >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
            
        max_candi_cnt = input_ids.size()[1]
        device = input_ids.device
        bsz = input_ids.size(0)
        
        if not generate:
            attention_mask = (input_ids != self.config.pad_token_id).long()

            if decoder_input_ids is None: 
                label_mask = labels > 0
                labels[~label_mask] = 0

                ones = torch.ones(bsz, 1).to(device)
                decode_start_input_ids = (ones * self.config.decoder_start_token_id).to(device).long()
                decoder_input_ids = torch.cat([decode_start_input_ids, labels], dim=1)
                
            elif labels is None:
                assert (decoder_input_ids[:, 0] == self.config.decoder_start_token_id).sum() == decoder_input_ids.size(0)
                labels = decoder_input_ids[:, 1:]
                label_mask = labels >= 0
                labels[~label_mask] = 0
            # NONONO --It is possible that we provide labels and decoder_inputs_ids together, usually they are not same, we may mask previous tokens in labels to focus on the generation of last token

            doc_mask = torch.arange(max_candi_cnt).unsqueeze(0).repeat([bsz, 1]).to(device)#[batch, max_candi_cnt]
            doc_mask = (doc_mask < candi_cnts.unsqueeze(-1)).view(bsz, -1) #[batch, max_candi_cnt]

            doc_mask = self.get_encode_atten_mask_given_target(doc_mask, labels).long() #[batch, topk, max_candi_cnt] 

            encoder_attention_mask = doc_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0] #[bsz, candi_cnt, dim]
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
                
            if decoder_attention_mask is not None:
                decoder_attention_mask = None

        # Decode
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError(
                f"You should provide one of decoder_input_ids or decoder_inputs_embeds"
            )
            
        if decoder_inputs_embeds is None and emb:
            #provide decoder_input_ids directly. otherwise we will provide decoder_input_embs for step-by-step prediction.
            target_input_ids = decoder_input_ids[:, 1:]# - self.config.origin_vocab_size
            target_embs = torch.gather(hidden_states, 1, target_input_ids.unsqueeze(-1).repeat([1,1,hidden_states.size(-1)]))#[batch, topk, dim]
            decoder_inputs_embeds = torch.cat([self.decoder.embed_tokens(decoder_input_ids[:, :1]), target_embs], dim=1)
            decoder_input_ids=None
 
            if decoder_attention_mask is not None:
                decoder_attention_mask = None
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]#[batch, topk, dim]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            sequence_output = sequence_output.to(self.encoder.first_device)#(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        dim = hidden_states.size(-1)
        lm_logits = torch.bmm(sequence_output, torch.transpose(hidden_states, 1, 2).contiguous()) / np.sqrt(dim)
        
        if self.training and labels is not None and loss:
            if ideal:
                loss = self.forward_post(labels, lm_logits[:, :-1, :], encoder_attention_mask.bool()[:, :-1, :], self.config.tmp, max_candi_cnt, left_doc_probs, candi_cnts, ideal, pre_list_cnt)
            else:
                loss = self.forward_post(labels, lm_logits, encoder_attention_mask.bool(), self.config.tmp, max_candi_cnt, left_doc_probs, candi_cnts, ideal, pre_list_cnt)
        else:
            loss = None
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=[sequence_output], #decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor = None,
        **kwargs,
    ):
        if 'max_new_tokens' not in kwargs:
            max_new_tokens = self.config.max_length
        else:
            max_new_tokens = kwargs['max_new_tokens']
            
        assert 'candi_cnts' in kwargs
        candi_cnts = kwargs['candi_cnts']
        
        if 'tmp' not in kwargs:
            tmp = 1
        else:
            tmp = kwargs['tmp']
            
        if 'do_sample' in kwargs:
            sample = kwargs['do_sample']
        else:
            sample = False
        
        bsz, max_candi_cnt = input_ids.size()[:2]
        all_max_candi_cnt = max_candi_cnt
        device = input_ids.device
        
        attention_mask = (input_ids != self.config.pad_token_id).long()
        
        doc_mask = torch.arange(all_max_candi_cnt).unsqueeze(0).repeat([bsz, 1]).to(device)
        doc_mask = (doc_mask < candi_cnts.unsqueeze(-1)).view(bsz, -1) #[batch, max_candi_cnt]
        
        ones = torch.ones(bsz, 1).to(device)
        decode_input_ids = (ones * self.config.eos_token_id).to(device).long()
        decode_attention_mask = ones.detach().long()
        decode_input_embs = self.decoder.embed_tokens(decode_input_ids)
        
        action_probs = None
        past_key_values = None
        decoder_hidden_states = None
        decoder_attentions = None
        cross_attentions = None
        encoder_last_hidden_state = None
        encoder_hidden_states = None
        encoder_attentions = None
        encoder_outputs = None
        
        for i in range(max_new_tokens):
            outputs = self(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                encoder_attention_mask=doc_mask.long(), # if doc_mask is not None else None,
                decoder_inputs_embeds=decode_input_embs,
                decoder_attention_mask=decode_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                generate=True,
            )
            logits = outputs.logits[:, i, :] / tmp #[batch, max_q_cnt*max_candi_cnt]
            past_key_values = outputs.past_key_values
            decoder_hidden_states = outputs.decoder_hidden_states
            decoder_attentions = outputs.decoder_attentions
            cross_attentions = outputs.cross_attentions
            encoder_last_hidden_state = outputs.encoder_last_hidden_state #[batch, candi_cnt, dim]
            encoder_hidden_states = outputs.encoder_hidden_states
            encoder_attentions = outputs.encoder_attentions

            encoder_outputs = [encoder_last_hidden_state, encoder_hidden_states, encoder_attentions]

            #mask invalid candi doc # this 'invalid' also consider previous selected docs
            
            if doc_mask is not None:
                logits[:, :all_max_candi_cnt] = logits[:, :all_max_candi_cnt].masked_fill_((~doc_mask).bool(), -torch.inf)
                logits[:, all_max_candi_cnt:] = -torch.inf
            #mask generated candi doc | with the previous step, we don't need this step.
         
            probs = F.softmax(logits, dim=-1) # [batch, max_q_cnt*max_candi_cnt]
            #mask invalid candi doc
            if doc_mask is not None:
                probs[:, :all_max_candi_cnt] = probs[:, :all_max_candi_cnt].masked_fill_((~doc_mask).bool(), 0.0)
                probs[:, all_max_candi_cnt:] = 0.0
            # #mask generated candi doc
            if sample:
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.max(probs, dim=1, keepdim=True)[1].long()

            next_token_embs = torch.gather(encoder_last_hidden_state, 1, next_tokens.unsqueeze(-1).repeat([1,1,encoder_last_hidden_state.size(-1)])) #[batch, 1, dim]
            act_prob = torch.gather(probs, 1, next_tokens) #[batch, 1]
            
            #update invalid doc
            doc_mask = self.mask_selected_docs(doc_mask.long(), next_tokens, 0).bool()

            if action_probs is None:
                generated_tokens = next_tokens 
                action_probs = act_prob
            else:
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
                action_probs = torch.cat([action_probs, act_prob], dim=1)

            decode_input_embs = torch.cat([decode_input_embs, next_token_embs], dim=1)
            decode_attention_mask = torch.cat([decode_attention_mask, ones], dim=1).long()
            
        return generated_tokens #[batch, topk]
        

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
