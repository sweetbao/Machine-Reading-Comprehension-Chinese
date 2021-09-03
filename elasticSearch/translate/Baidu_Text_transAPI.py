# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5


def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


def translate(from_lang, to_lang, query):
# Set your own appid/appkey.
    appid = '20210430000809347'
    appkey = 'Y2vGe1AUtUhdqyJ0TR9n'

    # # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    # from_lang = 'en'
    # to_lang =  'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    #query = 'Hello World! This is 1st paragraph.'

    # Generate salt and sign


    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    return result['trans_result'][0]['dst']


if __name__ == '__main__':
    re = translate('zh','en','第十条侦查人员讯问犯罪嫌疑人时，可以对讯问过程进行录音、录像;对可能判处无期徒刑、死刑或者其他重大罪行的，讯问过程应当录音录像。侦查人员应当将讯问过程的录音、录像告知犯罪嫌疑人，并在讯问笔录中写明。')
    rere = translate('en', 'zh', re)
    print(re)
    print(rere)