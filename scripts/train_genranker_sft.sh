rank_name="flan-t5-base"

WORK_ROOT="."
PLM_ROOT=".."
DATA_DIR="${WORK_ROOT}/all_dataset"
MODEL_DIR="${PLM_ROOT}/plms"

PRETRAINED_MODEL_RANK=${MODEL_DIR}/${rank_name}

dataset="wikipassageqa"
# selfdec=0
sub_type="golden" # "golden" "selfdec"

if [ $dataset = "wikiasp" ]; 
then
    if [ $sub_type = "golden" ]; 
    then
        TRAIN_FILE="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.qcatsubqqa_test.sft.ranker.jsonl"
        TRAIN_RETRIEVE_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.retrieve.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_RETRIEVE_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.retrieve.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_RETRIEVE_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.retrieve.qcatsubqqa_test.sft.ranker.jsonl"
    else
        TRAIN_FILE="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.llamadec.longq.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.llamadec.longq.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.decomposed.longq.qcatsubqqa_test.sft.ranker.jsonl"
        TRAIN_RETRIEVE_FILES="${DATA_DIR}/wikiasp/wikiasp_merge//wikiasp_merge.llamadec.longq.retrieve.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_RETRIEVE_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.llamadec.longq.retrieve.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_RETRIEVE_FILES="${DATA_DIR}/wikiasp/wikiasp_merge/wikiasp_merge.llamadec.longq.retrieve.qcatsubqqa_test.sft.ranker.jsonl"
    fi

    TOPK=10
    SUBQ=7
    PER_SUBQ=7
    MAX_DOC=293
    
elif [ $dataset = 'wikipassageqa' ]; 
then
    if [ $sub_type = "golden" ]; 
    then
        TRAIN_FILE="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.decomposed.qcatsubqqa_train.clean.sft.ranker.jsonl"
        EVAL_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.decomposed.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.decomposed.qcatsubqqa_test.sft.ranker.jsonl"
        TRAIN_RETRIEVE_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.retrieve.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_RETRIEVE_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.retrieve.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_RETRIEVE_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.retrieve.qcatsubqqa_test.sft.ranker.jsonl"
    else
        TRAIN_FILE="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.llamadec.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.llamadec.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.llamadec.qcatsubqqa_test.sft.ranker.nogold.new.jsonl"
        TRAIN_RETRIEVE_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.llamadec.retrieve.qcatsubqqa_train.sft.ranker.jsonl"
        EVAL_RETRIEVE_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.llamadec.retrieve.qcatsubqqa_dev.sft.ranker.jsonl"
        TEST_RETRIEVE_FILES="${DATA_DIR}/wikipassageqa/wikipassageqa_2source.llamadec.retrieve.qcatsubqqa_test.sft.ranker.nogold.jsonl"
    fi
    TOPK=10
    SUBQ=8
    PER_SUBQ=8
    MAX_DOC=270

else
    echo "Invalid dataset"
    exit
fi

SAVE_DIR="saved_ckpt/fidranker"
BATCH=1 #2
ACCU_STEP=4
LR=5e-5
PRECISION="bf16" #"fp32" "bf16"
TOTAL_STEPS=50000
PROJECT_NAME="debug"
if [ $sub_type = "golden" ]; 
then
    EXPERIMENT_NAME="sft-trl-${dataset}-${rank_name}-nq-${BATCH}-${LR}-${PROJECT_NAME}"
else
    EXPERIMENT_NAME="sft-selfdec-${dataset}-${rank_name}-nq-${BATCH}-${LR}"
fi
D_MAX_LEN=300
Q_MAX_LEN=25
L2_WEIGHT=0.5
SEED=42
# TEXT_MAX_LEN=$((${TOPK}*${D_MAX_LEN}+${Q_MAX_LEN}+10))
TOKENIZERS_PARALLELISM=true

deepspeed train_genranker_sft.py \
    --output_dir ${SAVE_DIR}/${EXPERIMENT_NAME} \
    --optim adamw_torch \
    --tmp 0.1 \
    --max_grad_norm 1 \
    --weight_decay 0.01 \
    --learning_rate ${LR} \
    --rank_model_path ${PRETRAINED_MODEL_RANK} \
    --q_maxlength ${Q_MAX_LEN} \
    --d_maxlength ${D_MAX_LEN} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --test_data ${TEST_FILES} \
    --train_retrieve_path ${TRAIN_RETRIEVE_FILES} \
    --eval_retrieve_path ${EVAL_RETRIEVE_FILES} \
    --test_retrieve_path ${TEST_RETRIEVE_FILES} \
    --topk ${TOPK} \
    --max_subq_cnt ${SUBQ} \
    --max_subq_cnt_per ${PER_SUBQ} \
    --max_candi_cnt ${MAX_DOC} \
    --l2 ${L2_WEIGHT} \
    --num_train_epochs 20 \
    --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size ${BATCH} \
    --gradient_accumulation_steps ${ACCU_STEP} \
    --seed ${SEED} \
    --evaluation_strategy steps \
    --logging_steps 1 \
    --log_level warning \
    --load_best_model_at_end True \
    --metric_for_best_model div \
    --greater_is_better True \
    --save_total_limit 1 \
    --eval_steps 10 \
    --save_steps 10 \
    --dataloader_num_workers 4 \
    --bf16 True \
    --deepspeed deepspeed/dp.json \
    --distrub 1 \
    --wandb_project RichRAG \
    --wandb_name ${PROJECT_NAME} \
    --report_to wandb  > train.${dataset}.$sub_type.log 2>&1 &
   