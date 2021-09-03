#coding: utf-8
# 从训练集中选取数据使用BERT进行相关度二分类训练

import json
import random
import os
from tqdm import tqdm
import sys
from normalize import filter_tags
import csv
from rouge import Rouge
import jieba
import copy

P = [0.503, 0.2314, 0.1414, 0.1031, 0.0411]


def get_rougeL(a,b):
    if len(a) <= 0 or len(b) <= 0:
        return 0
    if a == '.':
        a += ' '
    if b == '.':
        b += ' '
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b, avg=True)  # a和b里面包含多个句子的时候用
    # rouge_score1 = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    # r1 = rouge_score["rouge-1"]
    # r2 = rouge_score["rouge-2"]
    rl = rouge_score["rouge-l"]

    return rl['f']


def prepare_dataset(filename):
    """Prepare the sentence pair task training dataset for bert"""
    with open(filename, 'r', encoding='utf-8') as csvfile:
        datasets = []
        reader = list(csv.DictReader(csvfile))
        for row in tqdm(reader):
            print(row)
            row_cut_true = " ".join(jieba.lcut(row['true_answer']))
            row_cut_pred = " ".join(jieba.lcut(row['pred_answer']))
            rouge = get_rougeL(row_cut_true, row_cut_pred)
            # credit_in = float(row['start_logit'])+float(row['end_logit'])
            for row2 in reader:
                # 取同一问题的两个候选答案 将rouge-L大的作为正例
                if row2['ques_id'][:7] == row['ques_id'][:7] and row2 != row:
                    row2_cut_true = " ".join(jieba.cut(row2['true_answer']))
                    row2_cut_pred = " ".join(jieba.cut(row2['pred_answer']))
                    rouge2 = get_rougeL(row2_cut_true, row2_cut_pred)
                    if rouge2 > rouge:
                        datasets.append([row['ques_id'], "not_entailment", row['question'], row['pred_answer']])
                        datasets.append([row2['ques_id'], "entailment", row2['question'], row2['pred_answer']])
                    else:
                        datasets.append([row['ques_id'], "entailment", row['question'], row['pred_answer']])
                        datasets.append([row2['ques_id'], "not_entailment", row2['question'], row2['pred_answer']])
            # csvfile.seek(0)
        return datasets


def write_tsv(output_path, datasets):
    with open(output_path, 'w', encoding='utf-8') as f:
        write_line = '\t'.join(['ques_id', 'question', 'pred_answer', 'label'])
        f.write(write_line + '\n')
        for i, data in enumerate(datasets):
            write_line = '\t'.join([filter_tags(data[0]), filter_tags(data[2]),
                                    filter_tags(data[3]), str(data[1])])
            f.write(write_line + '\n')



def main():
    if not os.path.exists('./retriever_data'):
        os.mkdir('./retriever_data')
    print('Start loading answer selecting file.')
    train_datasets = prepare_dataset('./data/rankdata.csv')
    random.shuffle(train_datasets)
    write_tsv('./retriever_data/train.tsv', train_datasets)
    print('Done with preparing answer dataset.')


if __name__ == "__main__":
    main()
