from MeCab import Tagger
import ipadic
from my_module.tools import get_genkei_list

NON_STOP_WORD = ['動詞', '名詞', '形容詞', '形容動詞']
STOP_JOUKEN = ['代名詞', '接尾', '非自立', '数']
STOP_WORD = ['する', 'いる', 'ある', 'なる', 'ない', 'できる', 'おり', '行う', 'ば', 'ら']

tagger = Tagger(ipadic.MECAB_ARGS)

# 感性語の抽出
with open('/workspace/dataset/data_src/w_senti_vocab.txt', 'r') as f:
    emotion_word_list = []
    for line in f:
        tmp_list = line.split()
        emotion_word_list.append(tmp_list[0])

emo_word_count_dict = {emo_w : 0 for emo_w in emotion_word_list}

normal_word_count_dict = {}
count = 0

with open('/workspace/dataset/data_src/wiki40b_with_emotion/train/wakati/wiki_40b_train_with_emotion.txt', 'r') as f:
    for line in f:
        genkei_list = get_genkei_list(line, tagger)
        window_count_word_list = []
        for word in genkei_list:
            tagger_list = tagger.parse(word).split('\n')[0].split('\t')[1].split(',')
            hinshi = tagger_list[0]
            jouken = tagger_list[1]
            genkei = tagger_list[6]
            if (hinshi in NON_STOP_WORD) and (jouken not in STOP_JOUKEN) and (genkei not in STOP_WORD):
                window_count_word_list.append(genkei)
        for word in window_count_word_list:
            if word in emotion_word_list:
                emo_word_count_dict[word] += 1
                # print("emo_word:{} -> {}times".format(word, emo_word_count_dict[word]))
            else:
                if word in normal_word_count_dict:
                    normal_word_count_dict[word] += 1
                else:
                    normal_word_count_dict[word] = 1
                # print("normal_word:{} -> {}times".format(word, normal_word_count_dict[word]))
        count += 1
        percent = count / 2337909 * 100
        print("{}%".format(percent))
        # if(count == 100):
        #     break

print("make dict done")

# 生成したdictを出現回数順にソート
emo_word_count_dict = sorted(emo_word_count_dict.items(), key=lambda x:x[1], reverse=True)
normal_word_count_dict = sorted(normal_word_count_dict.items(), key=lambda x:x[1], reverse=True)

with open('/workspace/dataset/data/frequency/emotion_word_freq.txt', 'w') as f:
    for emo_word_tuple in emo_word_count_dict:
        f.write("{}\n".format(emo_word_tuple))

with open('/workspace/dataset/data/frequency/normal_word_freq.txt', 'w') as f:
    for normal_word_tuple in normal_word_count_dict:
        f.write("{}\n".format(normal_word_tuple))

print("done")