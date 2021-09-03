import json
import csv
with open('./AnswerReranking/data/rankdata.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ques_id','question','true_answer','pred_answer','start_logit','end_logit','from_para'])
    with open("./data/dev-v2.0.json",'r') as load_f:
        load_dict = json.load(load_f)
        load_dict = load_dict['data']
        with open("./test1_output/nbest_predictions.json",'r') as pred_f:
            pred = json.load(pred_f)
        for dic in load_dict:
            count = 1
            paracount = 0
            for paragraph in dic['paragraphs']:
                paracount += 1
                for qa in paragraph['qas']:
                    ques_id = qa['id']
                    question = qa['question']
                    for ans in qa['answers']:
                        true_answer = ans['text']
                        for answer in pred[ques_id]:
                            pred_answer = answer['text']
                            start_logit = answer['start_logit']
                            end_logit = answer['end_logit']
                            new_id = ques_id[:-1]
                            writer.writerow([new_id+str(count),question,true_answer,pred_answer,start_logit,end_logit,paracount])
                            count += 1
