#!/bin/bash

task=all_pipline
data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["./data/msmarco-full/TREC_DL_2019/queries_2019/","./data/msmarco-full/TREC_DL_2020/queries_2020/","./data/msmarco-full/dev_queries/"]'
eval_qrel_path='["./data/msmarco-full/dev_qrel.json","./data/msmarco-full/TREC_DL_2019/qrel.json","./data/msmarco-full/TREC_DL_2019/qrel_binary.json","./data/msmarco-full/TREC_DL_2020/qrel.json","./data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-t5seq-aq

echo "task: $task"

model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint-120000
index_dir=$model_dir/index
out_dir=$model_dir/out

python -m torch.distributed.launch --nproc_per_node=2 -m ours_t5_pretrainer.evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --task=index \
    --encoder_type=t5seq_pretrain_encoder \
    --collection_path=$collection_path

python -m ours_t5_pretrainer.evaluate \
    --task=index_2 \
    --index_dir=$index_dir 

   python -m ours_t5_pretrainer.evaluate \
    --task=retrieve \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --encoder_type=t5seq_pretrain_encoder \
    --q_collection_paths=$q_collection_paths \
    --eval_qrel_path=$eval_qrel_path
