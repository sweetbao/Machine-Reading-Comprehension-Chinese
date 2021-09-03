#coding: utf-8

import json
from normalize import convert_to_squad, filter_tags
import copy


def prepare_test_squad(rank_file, preprocessed_file):
    test_datasets = []
    with open(rank_file, 'r', encoding='utf-8') as f: 
        datas = json.load(f)
    with open(preprocessed_file, 'r', encoding='utf-8') as f: 
        for lidx, line in enumerate(f):
            samples = json.loads(line.strip())
            for sample in samples:
                data = {}
                data['question_text'] = sample['question']
                data['qas_id'] = str(sample['question_id'])
                if data['qas_id'] in datas:
                    # data['doc'] = filter_tags(datas[data['qas_id']])[:500]
                    ## 使用natualio方法预测
                    paras = [filter_tags(para) for para in datas[data['qas_id']].split('##')]
                    if len('##'.join(paras)) <= 500:
                        data['doc'] = '##'.join(paras)
                    else:

                        passage = '##'.join(paras[:2])
                        passage += '##'
                        for para in paras[2:]:
                            passage += para.split('。')[0]
                        data['doc'] = passage[:500]
                else:
                    # print(data['qas_id'])
                    data['doc'] = ''
                if 'answers' in sample.keys():
                    data['orig_answer_text'] = sample['answers']
                test_datasets.append(copy.deepcopy(data))
    return test_datasets


if __name__ == "__main__":
    zhidao_test_datasets = prepare_test_squad('./data/zhidao_dev_rank_output.json',
                                                './data/zhidao.dev.local.json')
    search_test_datasets = prepare_test_squad('./data/search_dev_rank_output.json',
                                                './data/search.dev.local.json')
    squad_datas = convert_to_squad(zhidao_test_datasets+search_test_datasets)
    with open('./data/dureader_dev.json', 'w', encoding='utf-8') as f:
        json.dump(squad_datas, f, ensure_ascii=False)
