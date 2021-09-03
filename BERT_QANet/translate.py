#!/usr/bin/env python

'''
Target Languages: 
Albanian (sq)
Amharic (am)
Arabic (ar)
Armenian (hy)
Azerbaijani (az)
Basque (eu)
Belarusian (be)
Bengali (bn)
Bosnian (bs)
Bulgarian (bg)
Catalan (ca)
Cebuano (ceb)
Chichewa (ny)
Chinese (Simplified) (zh)
Chinese (Traditional) (zh-TW)
Corsican (co)
Croatian (hr)
Czech (cs)
Danish (da)
Dutch (nl)
English (en)
Esperanto (eo)
Estonian (et)
Filipino (tl)
Finnish (fi)
French (fr)
Frisian (fy)
Galician (gl)
Georgian (ka)
German (de)
Greek (el)
Gujarati (gu)
Haitian Creole (ht)
Hausa (ha)
Hawaiian (haw)
Hebrew (iw)
Hindi (hi)
Hmong (hmn)
Hungarian (hu)
Icelandic (is)
Igbo (ig)
Indonesian (id)
Irish (ga)
Italian (it)
Japanese (ja)
Javanese (jw)
Kannada (kn)
Kazakh (kk)
Khmer (km)
Korean (ko)
Kurdish (Kurmanji) (ku)
Kyrgyz (ky)
Lao (lo)
Latin (la)
Latvian (lv)
Lithuanian (lt)
Luxembourgish (lb)
Macedonian (mk)
Malagasy (mg)
Malay (ms)
Malayalam (ml)
Maltese (mt)
Maori (mi)
Marathi (mr)
Mongolian (mn)
Myanmar (Burmese) (my)
Nepali (ne)
Norwegian (no)
Pashto (ps)
Persian (fa)
Polish (pl)
Portuguese (pt)
Punjabi (pa)
Romanian (ro)
Russian (ru)
Samoan (sm)
Scots Gaelic (gd)
Serbian (sr)
Sesotho (st)
Shona (sn)
Sindhi (sd)
Sinhala (si)
Slovak (sk)
Slovenian (sl)
Somali (so)
Spanish (es)
Sundanese (su)
Swahili (sw)
Swedish (sv)
Tajik (tg)
Tamil (ta)
Telugu (te)
Thai (th)
Turkish (tr)
Ukrainian (uk)
Urdu (ur)
Uzbek (uz)
Vietnamese (vi)
Welsh (cy)
Xhosa (xh)
Yiddish (yi)
Yoruba (yo)
Zulu (zu)
'''

# add guid
# create back-translation for context without any answers

from google.cloud import translate
import json
import timeit
import time
import sys

def xlate_old():
    with open("data/train-v2.0-questions-arabic.txt", "r") as qfile:
    #with open("data_ss.txt", "r") as qfile:
        q = qfile.readlines()
        q_count = 0

    with open("data/train-last.json", "r") as jsonFile:
    #with open("data_ss.json", "r") as jsonFile:
        a = json.load(jsonFile)
        b = a["data"]
        for i in b:
            c = i["paragraphs"]
            for j in c:
                d = j["qas"]
                for k in d:
                    k["id"] = uuid.uuid4().hex
                    #text = k["question"]
                    k["question"] = q[q_count].strip()
                    q_count += 1


    with open("data/train-v2.0-q-en-ar-en.json", "w") as jsonFile:
    #with open("data_ss_new.json", "w") as jsonFile:
        json.dump(a, jsonFile)


def xlate(handle, text, lang):
    #print(u'Original: {}'.format(text))
    for src, tgt in lang:
        ans = handle.translate(text,
            source_language=src,
            target_language=tgt,
            format_="text")
        #print(u'Text: {}'.format(text))
        #print(u'Translation: {}'.format(ans['translatedText']))
        text = ans['translatedText']
    #print(u'Translation: {}'.format(text))
    #print("########")
    return text

def merge_final():
    #with open("data/train-last.json", "r") as jsonFile:
    with open("data_ss_new.json", "r") as f1:
        a1 = json.load(f1)
        b1 = a1["data"]

    with open("data_ss.json", "r") as f2:
        a2 = json.load(f2)
        b2 = a2["data"]


    #with open("data/train-v2.0-q-en-ar-en.json", "w") as jsonFile:
    with open("aaaaa.json", "w") as jsonFile:
        for i in b1:
            json.dump(i, jsonFile)

        for j in b2:
            json.dump(j, jsonFile)


if __name__ == '__main__':
    print("Start test ...")
    c1, c2 = 0, 0

    start_time = timeit.default_timer()
    outfile = open("data/train-v2.0-questions-arabic.txt", 'a')

    # Instantiates a client
    translate_client = translate.Client()

    # The target language
    lang = [('en','ar'),('ar','en')]
    text = ""
    with open("data/train-v2.0-questions.txt", "r") as infile:
        for line in infile:
            #if c1 > 5000:
            #    break
            text += line
            c1 += 1
            if c1 % 15 == 0:
                out_text = xlate(translate_client, text, lang)
                outfile.write(out_text.encode('utf-8'))
                #outfile.write("\n")
                text = ""
                c2 += 1
                if c2%20 == 0:
                    print("Lines translated: ", c1)
                    '''
                    print "Sleeping",
                    for i in xrange(30):
                        sys.stdout.write('.')
                        sys.stdout.flush()
                        time.sleep(1)
                    print ""
                    '''

    outfile.close()

    print("End test.")
    print("Total time: ", timeit.default_timer() - start_time)
