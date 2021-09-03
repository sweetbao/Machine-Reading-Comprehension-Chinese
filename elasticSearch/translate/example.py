# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
from imp import reload
import json

import time

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
APP_KEY = '3b58e01ff79d7015'
APP_SECRET = 'zsfw4munIHEC0Zfb7hE5hL9ygOFBtrSL'


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def connect(q):
    data = {}
    data['from'] = '源语言'
    data['to'] = '目标语言'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['vocabId'] = "您的用户词表ID"

    response = do_request(data)
    contentType = response.headers['Content-Type']
    # if contentType == "audio/mp3":
    #     millis = int(round(time.time() * 1000))
    #     filePath = "合成的音频存储路径" + str(millis) + ".mp3"
    #     fo = open(filePath, 'wb')
    #     fo.write(response.content)
    #     fo.close()
    # else:
    #     print(response.content)
    translation = json.loads(response.content.decode())['translation']
    return translation

if __name__ == '__main__':
    query = '第十条侦查人员讯问犯罪嫌疑人时，可以对讯问过程进行录音、录像;对可能判处无期徒刑、死刑或者其他重大罪行的，讯问过程应当录音录像。侦查人员应当将讯问过程的录音、录像告知犯罪嫌疑人，并在讯问笔录中写明。'
    translation = connect(query)
    re_translation = connect(translation[0])
    print("原文本：", query)
    print("中间文本：", translation[0])
    print("双向翻译后的文本：", re_translation[0])


