# coding=UTF-8

from elasticsearch import Elasticsearch
import csv
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from digits2chinese import to_chinese
from chinese import four_to_han
from translate.example import connect
from translate.Baidu_Text_transAPI import translate
import json
import os.path
import csv
from tqdm import tqdm
from clear import clear
import copy
import difflib

DOC_Count = 10


def insert_QAData(doc_name, id, count):
    csv_set = []
    with open(doc_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = csv.reader(csvfile)
        column = [row['答案'] for row in reader]
        json_set = []
        for (i,c) in enumerate(column):
            dataitem = ''
            for j in range(5):
                if i+j <len(column):
                    dataitem += column[i+j]
            dic_tmp = {"name": doc_name[2:], "para_id": count, "paragraph": dataitem}
            csv_set.append(dataitem)
            count += 1
            dicJson = json.dumps(dic_tmp, ensure_ascii=False)
            json_set.append(dicJson)

    with open(doc_name, 'r') as csvfile:
        rows = csv.reader(csvfile)
        with open('./output'+doc_name[1:], 'w') as wfile:
            writer = csv.writer(wfile)
            index = 0
            for row in rows:
                row = row[:-1]
                if index == 0:
                    index += 1
                    continue
                if index < len(csv_set):
                    row.append(csv_set[index-1])
                    index += 1
                    writer.writerow(row)
    def gendata():
        for j in json_set:
            j = eval(j)
            yield {
                "_index": "regulations",
                "name": doc_name[0:-5],
                "para_id": j["para_id"],
                "paragraph": [j["paragraph"]]
            }

    helpers.bulk(es, gendata())
    id += 1
    # csv_file = csv.writer(open('./output/docshenan.csv', 'w'))
    # for i in csv_set:
    #     csv_file.writerow([i])


def insert_json(doc_name, id):
    with open('./regulations/'+doc_name, 'rb') as csv_in:
        with open('./regulations/c'+doc_name, "w", encoding="utf-8") as csv_temp:
            for line in csv_in:
                if not line:
                    break
                else:
                    line = line.decode("utf-8", "ignore")
                    #line = str(line).splitlines(True)
                    for l in line:
                        csv_temp.write(str(l).replace('\\n', ''))
    json_set = []
    csv_set = []
    count = 1
    if doc_name != ".DS_Store" and doc_name != "c.DS_Store":
        with open('./regulations/c'+doc_name, 'r') as f:
            li = f.readlines()
            for row in li:
                samples = json.loads(row)
                for sample in samples:
                    if sample["正文"] != "" and sample["正文"] != " ":
                        #datalist = sample["正文"].split(' ')
                        datalist = clear(sample["正文"])
                        for dataitem in datalist:
                            if len(dataitem) < 30:
                                continue
                            csv_set.append(dataitem)
                            dic_tmp = {"name": doc_name, "para_id": count, "paragraph": dataitem}
                            dicJson = json.dumps(dic_tmp, ensure_ascii=False)
                            json_set.append(dicJson)
                            count += 1
        f.close()
        os.remove('./regulations/c'+doc_name)
    else:
        os.remove('./regulations/'+doc_name)

    def gendata():
        for j in json_set:
            j = eval(j)
            yield {
                "_index": "regulations",
                "name": doc_name[0:-5],
                "para_id": j["para_id"],
                "paragraph": [j["paragraph"]]
            }

    helpers.bulk(es, gendata())
    id += 1
    csv_file = csv.writer(open('./output/docs.csv', 'w'))
    for i in csv_set:
        csv_file.writerow([i])
    # dataset = []
    # for line in csv_file:
    return id, count


def insert(doc_name,id):
    with open('./regulations/'+doc_name, 'r') as f1:
        list1 = f1.readlines()
    for i in range(0, len(list1)):
        list1[i] = list1[i].rstrip('\n')
    string1 ="".join(list1)
    last1 = 0
    last2 = 0
    dic = {"title":doc_name,"paragraphs":[]}
    para_id = 1
    json_set = []
    for i in range(1,300):
        if i<101 or i % 100 == 0:
            str_cn = to_chinese(i)
        else:
            str_cn = four_to_han(i)
        last1 = last2
        last2 = string1.find("第"+str_cn+"条", last2, len(string1))
        if(last2 == -1):
            break
        else:
            if i==1:
                continue
            dic_tmp = {"name":doc_name,"para_id":para_id,"paragraph":string1[last1:last2]}
            dicJson = json.dumps(dic_tmp, ensure_ascii=False)
            json_set.append(dicJson)
            #es.index(index="regulations", doc_type="法律法规", id=id, body=dicJson)
            #dic["paragraphs"].append(dic_tmp)
            para_id += 1
            id +=1
    dic_tmp = {"name":doc_name,"para_id":para_id,"paragraph":string1[last1:len(string1)]}
    dicJson = json.dumps(dic_tmp, ensure_ascii=False)
    json_set.append(dicJson)
    #es.index(index="regulations", doc_type="法律法规", id=id, body=dicJson)

    #helpers.bulk(client=es, actions=json_set)
    def gendata():
        for j in json_set:
            j = eval(j)
            yield {
                "_index": "regulations",
                "name": doc_name[0:-4],
                "para_id":j["para_id"],
                "paragraph":[j["paragraph"]]
            }

    helpers.bulk(es, gendata())
    id +=1
    if f1:
        f1.close()
    return id


def query(question, answer):
    query_list = es.indices.analyze(index="regulations",body={'text':question,'analyzer':"ik_max_word"})
    query1 = {
        "query": {
            "bool": {
                "should": [
                    # {"match": {"paragraph": "发明创造"}},
                    # {"match": {"paragraph": "报酬"}}
                ]
            }
        }
    }
    for q in query_list['tokens']:
        query1["query"]["bool"]["should"].append({"match": {"paragraph": q['token']}})

    #,filter_path="hits.hits._source.paragraphs"
    es.indices.refresh(index="regulations")
    query = es.search(index="regulations", filter_path="hits", scroll='5m', body=query1)
    value = query["hits"]["hits"]
    results = []
    if len(value)>DOC_Count:
        for i in range(DOC_Count):
            para_score = value[i]['_score']
            dataItem = {}
            if in_answer(value[i]['_source']['paragraph'],answer,para_score):
                dataItem['is_selected'] = True
            else:
                dataItem['is_selected'] = False
            dataItem['title'] = value[i]['_source']['name']
            dataItem['paragraphs'] = value[i]['_source']['paragraph']
            results.append(dataItem)
        return results
    else:
        for i in range(len(value)):
            para_score = value[i]['_score']
            dataItem = {}
            if in_answer(value[i]['_source']['paragraph'], answer,para_score):
                dataItem['is_selected'] = True
            else:
                dataItem['is_selected'] = False
            dataItem['title'] = value[i]['_source']['name']
            dataItem['paragraphs'] = value[i]['_source']['paragraph']
            results.append(dataItem)
        return results
    # es查询出的结果第一页
    #results = query['hits']['hits']
    # es查询出的结果总量
    #print(query['__len__'])
    # if 'total' in query:
    #     #print('true')
    #     total = query['hits']['total']
    # else:
    #     #print('false')
    #     total = {'value':1}


def get_name(rootdir):
    name_list = []
    for parent, dirnames, filenames in os.walk(rootdir):
        # for dirname in dirnames:
        #     print("Parent folder:", parent)
        #     print("Dirname:", dirname)
        for filename in filenames:
            name_list.append(filename)
    return name_list

#def build_dataset(paragraphs, ):


def in_answer(para, answer, score):
    '''添加判断该条段落是否出现在答案中'''
    return answer in para


def get_answer(para, answer):
    true_start = answer[0:2]
    true_end = answer[-2:]
    start_score = 0
    end_score = 0
    new_start = 0
    new_end = len(para)-3
    for (i,p) in enumerate(para):
        if i == len(para)-1:
            break
        tmp_s_score = get_simi_score(para[i:i+2], true_start)
        tmp_e_score = get_simi_score(para[i:i+2], true_end)
        if tmp_s_score>start_score:
            start_score = tmp_s_score
            new_start = i
        if tmp_e_score>end_score:
            end_score = tmp_e_score
            new_end = i
    return para[new_start:new_end+2]


def get_simi_score(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).quick_ratio()


if __name__ == '__main__':
    es = Elasticsearch(hosts="127.0.0.1", port=9200, timeout=200)
    if (es.indices.exists(index='regulations') is not True):
        es.indices.create(index='regulations')
    else:
        es.indices.delete(index="regulations")
        es.indices.create(index='regulations')
    name_list = get_name('./regulations')
    id = 1
    count = 1
    for name in name_list:
        id, count = insert_json(name,id)
        #id+=1

    insert_QAData('./河南检务公开数据.csv', id, count)

    def gen_json(doc_name):
        dataset = []
        csv_file = csv.reader(open(doc_name, 'r'))
        for line in csv_file:
            result = query(line[1], line[2])
            data = {}
            data['documents'] = result
            data['question'] = line[1]
            data['answers'] = [line[2]]
            data['question_type'] = 'DESCRIPTION'
            data['question_id'] = line[0]
            data['fact_or_opinion'] = 'FACT'
            dataset.append(data)
        return dataset

    whole_set = []
    whole_set += gen_json('ques_test.csv')
    whole_set += gen_json('./output/河南检务公开数据.csv')

        #继续写入双向翻译后的数据
        #翻译回来后的文档需要重新寻找一次答案
        # result_new = copy.deepcopy(result)
        # ans_new = data['answers']
        # for re in result_new:
        #     query_trans = re['paragraphs']
        #     translation = translate('zh', 'en', query_trans)
        #     re_translation = translate('en', 'zh', translation)
        #     re['paragraphs'] = re_translation
        #     if(re['is_selected']):
        #         ans_new = get_answer(re_translation, data['answers'])
        # data_new = {}
        # data_new['documents'] = result_new
        # data_new['question'] = line[1]
        # data_new['answers'] = ans_new
        # data_new['question_type'] = 'DESCRIPTION'
        # data_new['question_id'] = line[0]
        # data_new['fact_or_opinion'] = 'FACT'
        # dataset.append(data_new)



    with open("./output/dataset.json", "w") as f:
        json.dump(whole_set, f,ensure_ascii=False)
    print("加载入文件完成...")

    '''还没有写问题类型、factoropinion'''



