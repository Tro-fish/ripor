import ujson 
import os 

# assume we handle with the case of decay = 2
score_to_early_score = {8: 0.5/0.75, 16: 0.75/0.875, 32: 0.875/1.0}

max_new_token = 32
factor = score_to_early_score[max_new_token]
print("max_new_token: ", max_new_token, "factor: ", factor)

root_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/"
source_example_path = os.path.join(root_dir, f"syn_sfn_sub_smtid_train_decay2/syn_sfn_qid_smtids_scores_{max_new_token}.train.json")
out_dir = os.path.join(root_dir, "syn_sfn_sub_smtid_train_decay2")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

target_example_path = os.path.join(out_dir, f"syn_sfn_qid_smtid_scores_{max_new_token}.train.json")
with open(source_example_path) as fin:
    with open(target_example_path, "w") as fout:
        for line in fin:
            example = ujson.loads(line)
            example["early_scores"] = [x*factor for x in example["scores"]]
            fout.write(ujson.dumps(example) + "\n")