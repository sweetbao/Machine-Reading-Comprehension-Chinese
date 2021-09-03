import json
import timeit
import uuid

def get_english_arabic_translation():
    with open("train-v2.0-questions-arabic.txt", "r") as arfile:
        arabic = arfile.readlines()
    

    with open("train-v2.0-questions-copy.txt", "r") as infile:
        for i, line in enumerate(infile):
            line = line.lower()
            if 'how' in line or 'which' in line or 'who' in line or \
                'what' in line or 'where' in line or \
                'why' in line or 'when' in line:
                continue
            print("##################")
            print("English:", line)
            print("Arabic:", arabic[i])



def question_type():
    who, when, what, where, why, which, how, other = \
        [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
    
    with open("final_dict.json", "r") as jsonfile:
        a = json.load(jsonfile)
        for k, v in a.items():
            q = v['question'].lower()
            if "who" in q:
                who[0] += 1
                who[1] += v['correct_ans']
            elif "when" in q:
                when[0] += 1
                when[1] += v['correct_ans']
            elif "what" in q:
                what[0] += 1
                what[1] += v['correct_ans']
            elif "where" in q:
                where[0] += 1
                where[1] += v['correct_ans']
            elif "why" in q:
                why[0] += 1
                why[1] += v['correct_ans']
            elif "which" in q:
                which[0] += 1
                which[1] += v['correct_ans']
            elif "how" in q:
                how[0] += 1
                how[1] += v['correct_ans']
            else:
                other[0] += 1
                other[1] += v['correct_ans']
                #print(q)

    who.append(who[1]/who[0] * 100)
    when.append(when[1]/when[0] * 100)
    what.append(what[1]/what[0] * 100)
    where.append(where[1]/where[0] * 100)
    why.append(why[1]/why[0] * 100)
    which.append(which[1]/which[0] * 100)
    how.append(how[1]/how[0] * 100)
    other.append(other[1]/other[0] * 100)


    print("who:", who)
    print("when:", when)
    print("what:", what)
    print("where:", where)
    print("why:", why)
    print("which:", which)
    print("how:", how)
    print("other:", other)

def create_dataset_based_on_que_type(outfilename=None, search_criteria=None):
    ans = {"version": "v2.0", "data": []}

    with open("data/search.dev.json", "r") as jsonFile:
        a = json.load(jsonFile)
        
        for i in a["data"]:
            for j in i["paragraphs"]:
                qas_list = j["qas"]
                new_qas_list = []
                for item in qas_list:
                    if 'what' in item['question'].lower():
                        new_qas_list.append(item)
                        #print(item['question'].lower())
                print("##########################################")
                print("Original qas list:", len(qas_list))
                j["qas"] = new_qas_list[:]
                print("     New qas list:", len(new_qas_list))
                print("Modified qas list:", len(j["qas"]))

                #for item in qas_list:
                    #print(item['question'].lower())

    with open("data/dev-what.json", "w") as jsonFile:
        json.dump(a, jsonFile)

def calculate_change():
    a1 = [77.38, 77.36, 74.61, 74.93, 71.07, 69.45, 55.43, 52.54]
    b1 = [76.27, 75.40, 74.12, 73.10, 69.75, 66.18, 57.60, 64.40]

    a = [80.34, 79.10, 77.59, 76.99, 75.37, 72.10, 68.40, 62.38]
    b = [79.26, 77.32, 77.15, 75.18, 73.46, 69.76, 69.87, 72.58]

    for i in range(len(a)):
        print("diff:", (b[i]-a[i])/a[i])


def get_raw_scores(dataset, preds):
  count = 0
  min_avg, max_avg = 100, 0
  exact_scores = {}
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        #print(gold_answers)
        #length = sum([len(x.split()) for x in gold_answers])/len(gold_answers)
        if 'how' in qa['question'].lower() or \
          'which' in qa['question'].lower() or \
          'who' in qa['question'].lower() or \
          'what' in qa['question'].lower() or \
          'where' in qa['question'].lower() or \
          'why' in qa['question'].lower() or \
          'when' in qa['question'].lower():
          continue
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
        print("question:", qa['question'])
        print("a_pred:", a_pred)
        print("gold:", gold_answers)
        print("####################")
        
  return exact_scores, f1_scores

if __name__ == '__main__':
    print("Start test ...")
    start_time = timeit.default_timer()

    get_english_arabic_translation()
    question_type()
    create_dataset_based_on_que_type()
    calculate_change()

    with open("data/search.dev.json", 'r') as fd:
        dataset_json = json.load(fd)
        dataset = dataset_json['data']
    with open("output/predictions.json", 'r') as fp:
        preds = json.load(fp)
    
    get_raw_scores(dataset, preds)
    
    print("End test.")
    print("Total time: ", timeit.default_timer() - start_time)
