import os 

import wandb
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from torch.utils.data.dataloader import DataLoader
import ujson
import torch
from copy import deepcopy

from .modeling.t5_generative_retriever_ours import (
    T5SeqPretrainEncoder
)
from .tasks.trainer import  CondDocID_DRTrainer, CondDocID_TrainingArgs

from .dataset.dataset import (
    TripleMarginMSEDataset,
    MarginMSEforT5SeqAQDataset,
    Seq2SeqForT5SeqAQDataset,
    LngKnpMarginMSEforT5SeqAQDataset,
    MarginMSEforPretrainDataset,
)
from .dataset.data_collator import (
    MarginMSEforT5SeqAQCollator,
    Seq2SeqForT5SeqAQCollator,
    LngKnpMarginMSEforT5SeqAQCollator,
    MarginMSEforPretrainCollator
)
from .arguments import ModelArguments, Arguments
from .losses.regulariaztion import RegWeightScheduler

def save_train_args(model_args, args):
    merged_args = {**asdict(model_args), **asdict(args)}
    out_dir = deepcopy(args.output_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        ujson.dump(merged_args, f, indent=4)

def main():
    parser = HfArgumentParser((ModelArguments, Arguments))
    model_args, args = parser.parse_args_into_dataclasses()

    # save args log to disk
    if args.local_rank <= 0:
        save_train_args(model_args, args)
    
    eval_dataset = None 
    eval_collator = None

    model = T5SeqPretrainEncoder.from_pretrained(args.pretrained_path)
    train_collator = MarginMSEforPretrainCollator(model_args.model_name_or_path, max_length=args.max_length)

    train_dataset = MarginMSEforPretrainDataset(
        dataset_path=args.teacher_score_path,
        document_dir=args.collection_path,
        query_dir=args.queries_path,
        docid_to_smtid_path=args.docid_to_smtid_path
    )
    
    if args.local_rank <= 0:
        print("load model from pretrained path = {}".format(args.pretrained_path))
        print("model: ", args.loss_type, "model_args: ", model_args if model_args is not None else "not use")
        print("sanity check dataloader: ")
        dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=False, collate_fn=train_collator)
        for i, batch in enumerate(dataloader):
            for k, v in batch.items():
                if "tokenized" in k:
                    print(f"{k}: ", v["input_ids"].shape)
                elif "qd_kwargs" in k:
                    print(f"{k}: ", v["input_ids"].shape)
                else:
                    print(f"{k}: ", v.shape)
            if i == 0:
                break
        dataloader = None

    training_args = CondDocID_TrainingArgs(
        output_dir=args.output_dir,
        do_train=True,
        tokenizer_dir=args.tokenizer_dir,
        do_eval=args.do_eval,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        disable_tqdm=False,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        save_total_limit=5,
        seed=2,
        remove_unused_columns=False,
        task_names=args.task_names,
        ln_to_weight=args.ln_to_weight,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=args.use_fp16,
        #bf16=False,
        #no_cuda=True
        #dataloader_num_workers=4
    )

    if training_args.local_rank <= 0:  # only on main process
        wandb.login()
        wandb.init(project=args.wandb_project_name, name=args.run_name)
    
    if args.local_rank <= 0:
        print("docid_to_smtids_path: ", args.docid_to_smtid_path)
        print("pretrained_path: ", args.pretrained_path)

    if args.model_type in ["t5_docid_gen_encoder"]:
        trainer = CondDocID_DRTrainer(
            model = model,
            train_dataset = train_dataset,
            data_collator = train_collator,
            args = training_args,
            eval_dataset = eval_dataset,
        )
        trainer.add_eval_data_collator(eval_collator)
        if args.loss_type in ["sparse_project_margin_mse", "sparse_project_pretrain_margin_mse"]:
            if training_args.max_steps > 0:
                trainer.add_reg_scheduler(RegWeightScheduler(lambda_=args.ln_to_weight["reg"], T=training_args.max_steps // 3))
                print("saturated steps for sparse reg = {}".format(training_args.max_steps // 3))
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    
    trainer.train()
    trainer.save_torch_model_and_tokenizer(train_collator.tokenizer)
        
if __name__ == "__main__":
    main()