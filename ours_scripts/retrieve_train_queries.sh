#!/bin/bash

# Variables
data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["./data/msmarco-full/TREC_DL_2019/queries_2019/","./data/msmarco-full/TREC_DL_2020/queries_2020/","./data/msmarco-full/dev_queries/"]'
eval_qrel_path='["./data/msmarco-full/dev_qrel.json","./data/msmarco-full/TREC_DL_2019/qrel.json","./data/msmarco-full/TREC_DL_2019/qrel_binary.json","./data/msmarco-full/TREC_DL_2020/qrel.json","./data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-t5seq-aq

# Step 1: Run retrieve_train_queries task
echo "Running retrieve_train_queries task"

model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
index_dir=$model_dir/index
out_dir=$model_dir/out/
pretrained_path=$model_dir/checkpoint-120000

python -m ours_t5_pretrainer.evaluate \
    --task=retrieve \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir  \
    --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
    --topk=100 \
    --encoder_type=t5seq_pretrain_encoder

# Step 2: Rerank for create_trainset
echo "Running rerank_for_create_trainset task"

root_dir=./experiments-full-t5seq-aq/
run_path=$root_dir/t5_docid_gen_encoder_0/out/MSMARCO_TRAIN/run.json
out_dir=$root_dir/t5_docid_gen_encoder_0/out/MSMARCO_TRAIN/
q_collection_path=./data/msmarco-full/all_train_queries/train_queries

python -m torch.distributed.launch --nproc_per_node=2 -m ours_t5_pretrainer.rerank \
    --task=rerank_for_create_trainset \
    --run_json_path=$run_path \
    --out_dir=$out_dir \
    --collection_path=$collection_path \
    --q_collection_path=$q_collection_path \
    --json_type=json \
    --batch_size=256

python -m ours_t5_pretrainer.rerank \
    --task=rerank_for_create_trainset_2 \
    --out_dir=$out_dir

# Step 4: Add relevant docids to the rerank output
echo "Running add_qrel_to_rerank_run task"

python ours_t5_pretrainer/aq_preprocess/add_qrel_to_rerank_run_for_pretrain.py