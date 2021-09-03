import jieba
import json
from collections import Counter
'''通过两者结构，预处理后的数据相比于原始数据增加了分词结果，
并且在每篇文档中增加了与问题最相关的段落“most_related_para”字段；
由于目前阅读理解框架都是基于Span抽取的，因此增加了“fake_answers”字段，表示伪造答案，
“answer_docs”字段表示伪造答案来自于哪一篇文档，
“answer_spans”字段表示伪造答案所在文档的位置信息，
“match_scores”表示伪造答案的评分值。'''

def get_fake_data(path, save_path):
    fin = open(save_path, "w")
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            # 导入原始数据
            samples = json.loads(line)
            for sample in samples:
                # 对answers和question进行分词
                sample["segmented_answers"] = [seg_word(answer) for answer in sample["answers"]]
                sample["segmented_question"] = seg_word(sample["question"])
                for doc in sample["documents"]:
                    # 对每个篇章的title和paragraphs进行分词
                    doc["segmented_title"] = seg_word(doc["title"])
                    doc["segmented_paragraphs"] = [seg_word(para) for para in doc["paragraphs"]]
                # 对原始数据求解伪造答案和其相关字段参数
                find_fake_answer(sample)
                fin.write(json.dumps(sample, ensure_ascii=False) + "\n")

def seg_word(text):
    text_temp = list(jieba.cut(text))
    return text_temp
def find_fake_answer(sample, max_length_answer=None):
    # 求解出每个文档中与答案最相关的段落
    for doc in sample['documents']:
        most_related_para = -1
        most_related_para_len = 999999
        max_related_score = 0
        # 将每个段落与答案集合进行分数计算
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):
            # 当答案集合不为空时，求出段落与答案集合的recall值；当答案集合为空时，不做计算
            if len(sample['segmented_answers']) > 0:
                related_score = metric_max_over_ground_truths(recall,
                                                              para_tokens,
                                                              sample['segmented_answers'])
            else:
                continue
            # 判断，如果计算得到的分数大于最大分数时，或者计算得到的分数等于最大分数，并且当前段落长度小于最相关段落长度时
            # 则修改最相关段落为当前段落，最大值为当前值，最相关段落长度为当前段落长度
            if related_score > max_related_score \
                    or (related_score == max_related_score
                        and len(para_tokens) < most_related_para_len):
                most_related_para = p_idx
                most_related_para_len = len(para_tokens)
                max_related_score = related_score
                # 最终保存下，每个文档与答案最相关的段落编号
        doc['most_related_para'] = most_related_para

    sample['answer_docs'] = []
    sample['answer_spans'] = []
    sample['fake_answers'] = []
    sample['match_scores'] = []

    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    # 得到答案词集合，为后面计算提供帮助
    answer_tokens = set()
    for segmented_answer in sample['segmented_answers']:
        answer_tokens = answer_tokens | set([token for token in segmented_answer])
    # 由于当前阅读理解模型，都是进行span抽取，因此需要保证答案必须在篇章中。
    # 下面求解每篇文档的最相关段落的最相关的span片段
    for d_idx, doc in enumerate(sample['documents']):
        # 如果答案未参考该篇文档，那么则跳过。
        if not doc['is_selected']:
            continue
        if doc['most_related_para'] == -1:
            doc['most_related_para'] = 0
        most_related_para_tokens = doc['segmented_paragraphs'][doc['most_related_para']]
        # 求解伪造答案时，使用贪婪方式进行计算
        # 起始位置：从前往后遍历，结束位置：从后往前遍历
        for start_tidx in range(len(most_related_para_tokens)):
            # 为了节约时间，若当前字符不在答案集合时，跳过
            if most_related_para_tokens[start_tidx] not in answer_tokens:
                continue
            # 为了节省时间,增加答案最长参数
            # 默认答案最大长度为整个整个段落，那么结束位置从段落最后一位开始遍历
            # 若设置答案最大长度，么结束位置从起始位置加上最大长度位置开始遍历
            if max_length_answer is None:
                answer_tokens_len = len(most_related_para_tokens)
            else:
                answer_tokens_len = min((max_length_answer + start_tidx), len(most_related_para_tokens))
            for end_tidx in range(answer_tokens_len - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
                # 当答案集合不为空时，求出span片段与答案集合的f1值；当答案集合为空时，不做计算
                if len(sample['segmented_answers']) > 0:
                    match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                                sample['segmented_answers'])
                else:
                    match_score = 0
                if match_score == 0:
                    break
                # 记录f1值最高的span片段，及其所在文档序号、分数和起始结束位置
                if match_score > best_match_score:
                    best_match_d_idx = d_idx
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_fake_answer = ''.join(span_tokens)
                    # 若伪造答案f1值不为0，将其相关参数加到样本字典中
    if best_match_score > 0:
        sample['answer_docs'].append(best_match_d_idx)
        sample['answer_spans'].append(best_match_span)
        sample['fake_answers'].append(best_fake_answer)
        sample['match_scores'].append(best_match_score)
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    # 分别计算预测序列与多个标准序列的指标, metric_fn可以为recall或者f1
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    # 对多个值,取最大
    return max(scores_for_ground_truths)

def recall(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[1]

def f1_score(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[2]
def precision_recall_f1(prediction, ground_truth):
    # 判断预测序列prediction是否为list型
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction
    # 判断标准序列ground_truth是否为list型
    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth
    # 计算预测序列和标准序列的相同词的个数
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 对其加和,表示两个序列共有多少个词相同
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    # 准确率为词相同除以预测序列词个数
    p = 1.0 * num_same / len(prediction_tokens)
    # 召回率为词相同除以标准序列词个数
    r = 1.0 * num_same / len(ground_truth_tokens)
    # f1为 2倍的准确率乘以召回率,除以准确率与召回率之和
    f1 = (2 * p * r) / (p + r)
    return p, r, f1

if __name__ == '__main__':
    get_fake_data('dataset.json', './result.json')
    # for line in sys.stdin:
    #     sample = json.loads(line)
    #     find_fake_answer(sample)
    #     print(json.dumps(sample, encoding='utf8', ensure_ascii=False))
