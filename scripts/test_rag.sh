reader_root='../plms'
reader_name='llama-2-13b-chat-hf'
# reader_name='vicuna-7b-v1.5'
rank_model_name='flan-t5-base'

data_baseroot="./all_dataset"
retrieve_root="${data_baseroot}/atlas_data/corpus_index/bge-base-en-v1.5"

dataset="wikipassageqa" #"wikipassageqa" "wikiasp"
sub_type="golden" #"golden" "selfdec"
suffix=""
if [ $dataset = "wikiasp" ];
then
    if [ $sub_type = "golden" ];
    then
        test_name='wikiasp_merge.decomposed.longq.qcatsubqqa_test.sft.ranker.jsonl'
        test_comretrieve_name='wikiasp_merge.decomposed.longq.retrieve.qcatsubqqa_test.sft.ranker.jsonl'
        test_retrieve_name='wikiasp_merge_decomposed.qcatsubqsubqa.longq.test.jsonl'
        MODEL_PATH="saved_ckpt/dpo/golden_dec_wikiasp_ckpt/model.safetensors"
        use_decompose=1
    else
        test_name='wikiasp_merge.llamadec.longq.qcatsubqqa_test.sft.ranker.jsonl'
        test_comretrieve_name='wikiasp_merge.llamadec.longq.retrieve.qcatsubqqa_test.sft.ranker.jsonl'
        test_retrieve_name='wikiasp_merge_decomposed.qcatsubqqa_llamadec_test_sft.jsonl'
        MODEL_PATH="saved_ckpt/dpo/selfdec_wikiasp_ckpt/model.safetensors"
        use_decompose=0
    fi
    data_root="${data_baseroot}/wikiasp/wikiasp_merge"
    MAX_CANDI=293
    max_subq_cnt=7
elif [ $dataset = 'wikipassageqa' ]; 
then
    if [ $sub_type = "golden" ];
    then
        test_name='wikipassageqa_2source.decomposed.qcatsubqqa_test.sft.ranker.nogold.jsonl'
        test_comretrieve_name='wikipassageqa_2source.retrieve.qcatsubqqa_test.sft.ranker.nogold.jsonl'
        test_retrieve_name='wikipassageqa_2source.decomposed.qcatsubqqa_test.sft.jsonl'
        MODEL_PATH="saved_ckpt/dpo/golden_dec_wikipsg_ckpt/model.safetensors"
        use_decompose=1
    else
        test_name='wikipassageqa_2source.llamadec.qcatsubqqa_test.sft.ranker.nogold.new.jsonl'
        test_comretrieve_name='wikipassageqa_2source.llamadec.retrieve.qcatsubqqa_test.sft.ranker.nogold.jsonl'
        test_retrieve_name='wikipassageqa_2source.decomposed.qcatsubqqa_llamadec_test_sft.jsonl'
        MODEL_PATH="saved_ckpt/dpo/selfdec_wikipsg_ckpt/model.safetensors"
        use_decompose=0
    fi
    data_root="${data_baseroot}/wikipassageqa"
    MAX_CANDI=270
    max_subq_cnt=8
else
    echo "Invalid dataset"
    exit
fi 
topk=10

python -u test_rag.py \
    --infer_batch 64 \
    --eval_batch 64 \
    --reader_model_path ${reader_root}/${reader_name} \
    --rank_model_path ${reader_root}/${rank_model_name} \
    --test_data ${data_root}/${test_name} \
    --test_comretrieve_data ${data_root}/${test_comretrieve_name} \
    --test_retrieve_data ${retrieve_root}/${test_retrieve_name} \
    --dataset $dataset \
    --retriever_n_context $MAX_CANDI \
    --model_path ${MODEL_PATH} \
    --topk $topk \
    --use_decompose $use_decompose \
    --debug_test_path "outputs/${dataset}-${sub_type}-${reader_name}${suffix}.jsonl" > eval_results/${dataset}-${sub_type}-${reader_name}${suffix}.log 2>&1 &