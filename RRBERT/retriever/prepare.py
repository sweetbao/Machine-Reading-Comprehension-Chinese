#coding: utf-8
# 从训练集中选取数据使用BERT进行相关度二分类训练

import json
import random
import os
from tqdm import tqdm
import sys
sys.path.append( '/Users/abaobao/Library/Application\ Support/JetBrains/PyCharm2020.1/scratches/BERT-Dureader-master/utils' )
from normalize import filter_tags


def prepare_dataset(filename):
    """Prepare the sentence pair task training dataset for bert"""
    with open(filename, 'r', encoding='utf-8') as f: 
        datasets = []
        for lidx, line in enumerate(tqdm(f)):
            samples = json.loads(line.strip())
            for sample in samples:
                qid = sample['question_id']
                question = sample['question']
                if not len(sample['match_scores']):
                    continue
                if sample['match_scores'][0] < 0.7:
                    continue
                if not len(sample['answer_docs']):
                    continue
                if sample['answer_docs'][0] >= len(sample['documents']):
                    continue
                doc = sample['documents'][int(sample['answer_docs'][0])]
                related_para = doc['paragraphs'][int(doc['most_related_para'])].replace('\n', '')
                datasets.append([1, question, related_para])
                for i in range(len(doc['paragraphs'])):
                    if i != int(doc['most_related_para']):
                        irrelated_para = doc['paragraphs'][i].replace('\n', '')
                        datasets.append([0, question, irrelated_para])
    return datasets


def write_tsv(output_path, datasets):
    with open(output_path, 'w', encoding='utf-8') as f: 
        for i, data in enumerate(datasets):
            write_line = '\t'.join([str(data[0]), str(i), str(i), filter_tags(data[1]), 
                                    filter_tags(data[2])])
            f.write(write_line + '\n')



def main():
    if not os.path.exists('./retriever_data'):
        os.mkdir('./retriever_data')
    print('Start loading preprocessed train json file.')
    search_datasets = prepare_dataset('/home/wdbao/tmp/pycharm_project_631/BERT-Dureader-master/optiBERT_Dureader/BERT_QANet/data/search.train.local.json')
    zhidao_datasets = prepare_dataset('/home/wdbao/tmp/pycharm_project_631/BERT-Dureader-master/optiBERT_Dureader/BERT_QANet/data/zhidao.train.local.json')
    train_datasets = search_datasets + zhidao_datasets
    random.shuffle(train_datasets)
    write_tsv('./retriever_data/train.tsv', train_datasets)
    search_dev_datasets = prepare_dataset('/home/wdbao/tmp/pycharm_project_631/BERT-Dureader-master/optiBERT_Dureader/BERT_QANet/data/search.dev.local.json')
    zhidao_dev_datasets = prepare_dataset('/home/wdbao/tmp/pycharm_project_631/BERT-Dureader-master/optiBERT_Dureader/BERT_QANet/data/zhidao.dev.local.json')
    dev_datasets = search_dev_datasets + zhidao_dev_datasets
    write_tsv('./retriever_data/dev.tsv', dev_datasets)
    print('Done with preparing training dataset.')


if __name__ == "__main__":
    main()
