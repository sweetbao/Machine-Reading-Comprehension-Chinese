import re
from digits2chinese import to_chinese
from chinese import four_to_han

def clear(val_list):
    data = []
    last1 = 0
    last2 = 0
    for i in range(1, 2000):
        if i < 101 or i % 100 == 0:
            str_cn = to_chinese(i)
        else:
            str_cn = four_to_han(i)
        last1 = last2
        last2 = val_list.find("第" + str_cn + "条", last2, len(val_list))
        if (last2 == -1):
            break
        else:
            if i == 1:
                continue
            else:
                jlast1 = 0
                jlast2 = 0
                for j in range(1,10):
                    j_cn = to_chinese(j)
                    jlast1 = jlast2
                    jlast2 = val_list[last1:last2].find("(" + j_cn + ")", jlast2, last2-last1)
                    if (jlast2 == -1):
                        jlast2 = val_list[last1:last2].find("（" + j_cn + "）", jlast2, last2 - last1)
                        if (jlast2 == -1):
                            string_tmp = val_list[last1:last2]
                            string_tmp = string_tmp.replace("\\n", "")
                            string_tmp = string_tmp.replace("\n", "")
                            string_tmp = string_tmp.replace("\u2002", "")
                            string_tmp = string_tmp.replace("\u3000", "")
                            data.append(string_tmp)
                            break
                        else:
                            string_tmp = val_list[jlast1:jlast2]
                            string_tmp = string_tmp.replace("\\n", "")
                            string_tmp = string_tmp.replace("\n", "")
                            string_tmp = string_tmp.replace("\u2002", "")
                            string_tmp = string_tmp.replace("\u3000", "")
                            data.append(string_tmp)
    return data


def not_use():
    doc_name = '厦门检察网数据.txt'
    with open('./regulations/' + doc_name, 'r') as f:
        with open('./output/' + doc_name[:-4]+'.txt', 'w') as newf:
            li = f.readlines()
            for row in li:
                results = clear(row)
                for result in results:
                    newf.write(result)