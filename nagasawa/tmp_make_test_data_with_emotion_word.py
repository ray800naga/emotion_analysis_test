from my_module.tools import has_emotion_word
from transformers import BertJapaneseTokenizer
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
with open('data_src/w_senti_vocab.txt', 'r') as f:
    emotion_word_list = []
    for line in f:
        tmp_list = line.split()
        emotion_word_list.append(tmp_list[0])
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
with open('data_src/wiki40b_one_stc/wiki_40b_train_one_stc.txt', 'r') as f_read:
    with open('data_src/wiki40b_with_emotion/wiki_40b_train_with_emotion.txt', 'w') as f_write:
        count = 0
        line_count = 0
        for line in f_read:
            tokens = tokenizer.tokenize(line)
            if has_emotion_word(tokens, emotion_word_list):
                count = count + 1
                f_write.write(line)
            line_count += 1
            print('{}%'.format(line_count / 12330278 * 100))
print(count)