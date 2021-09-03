import csv
import numpy as np
from tqdm import tqdm
import json


def get_mix_credit():
    P = [0.503, 0.2314, 0.1414, 0.1031, 0.0411]# 文档先验概率
    credits = {}
    e = 2.718281828459
    with open('./data/QNLI/results/pred_credit.csv', 'r') as rank_credit_file:
        csv_reader = csv.reader(rank_credit_file)
        rank_dict = {}
        for row1 in csv_reader:
            if row1[0] not in rank_dict:
                rank_dict[row1[0]] = row1[1]
        with open('./data/rankdata.csv', 'r') as self_credit_file:
            csv_reader2 = csv.DictReader(self_credit_file)
            exp = []
            for row2 in csv_reader2:
                self_credit = 1 + float(row2['start_logit']) + float(row2['end_logit'])
                rank_credit = float(rank_dict[row2['ques_id']])+1
                doc_dict = P[int(row2['from_doc'])]
                tmp = self_credit * doc_dict
                credits[row2['ques_id']] = rank_credit * (e**tmp)
                exp.append(e**tmp)
            for i in credits:
                credits[i] /= sum(exp)
    return credits


if __name__ == "__main__":
    credit = get_mix_credit()
    new = {}
    for key in credit:
        last_key = key[:-1]+'1'
        if last_key not in new:
            new[last_key] = {'key': key, 'credit': credit[key]}
        else:
            if credit[key] > new[last_key]['credit']:
                new[last_key] = {'key': key, 'credit': credit[key]}
    final_answer = {}
    json1 = {}
    with open('./data/rankdata.csv', 'r') as self_credit_file:
        csv_reader2 = csv.DictReader(self_credit_file)
        for row in tqdm(csv_reader2):
            json1[row['ques_id']] = row['pred_answer']
    for key in new:
        final_answer[key] = json1[new[key]['key']]
    with open("./predictions.json", "w") as dump_f:
        json.dump(final_answer, dump_f)
