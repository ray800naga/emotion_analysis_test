from transformers import BertJapaneseTokenizer, BertModel
from torch.utils.data import Dataset
import torch
import os

# bertでトークナイズされたトークン列のなかに感性語が含まれているか否かを確認
def has_emotion_word(tokens, emotion_word_list):
    for emo_w in emotion_word_list:
        for token in tokens:
            if emo_w == token:
                print(emo_w) # for debug
                return True
    return False

# 取得したバッチからデータセットを生成
def get_dataset_from_batch(batch, last_hidden_state, file_count, batch_count, output_name_head):
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    emotion_word_dict = get_emotion_word_dict()
    batch_len = len(batch['input_ids'])
    # print("batch_num: ", batch_num)
    with open(output_name_head + "{:0>4}_{:0>4}.txt".format(file_count, batch_count), 'w') as f:
        for i in range(batch_len):
            # input_ids には1文のid列が格納されている。
            tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
            del tokens[0]   # [CLS]を削除
            del last_hidden_state[i][0]    # [CLS]に対応する出力も削除
            sep_idx = tokens.index('[SEP]') # [SEP]のindexを取得
            del tokens[sep_idx:]    # [SEP]以降を削除
            del last_hidden_state[i][sep_idx:] # 対応する出力も削除
            if len(tokens) != len(last_hidden_state[i]):
                print("error")
                continue
            emotion_word_idx_list = get_emotion_word_idx(tokens)
            for emotion_word_idx in emotion_word_idx_list:
                emotion_vector = emotion_word_dict[tokens[emotion_word_idx]]
                # print("{} : {}".format(tokens[emotion_word_idx], emotion_vector))
                f.write("{}\t{}\n".format(last_hidden_state[i][emotion_word_idx], emotion_vector))
    

# tokensから感性語のインデックスのリストを返す。
def get_emotion_word_idx(tokens):
    idx_list = []
    emotion_word_dict = get_emotion_word_dict()
    for idx, token in enumerate(tokens):
        for emotion_word in emotion_word_dict.keys():
            if token == emotion_word:
                idx_list.append(idx)
                break
    # print('idx_list:', idx_list)
    return idx_list



# 感性語がKey, 感情ベクトルがValueのdictを取得
def get_emotion_word_dict():
    with open('/workspace/emotion_analysis_test/nagasawa/data_src/w_senti_vocab.txt', 'r') as f:
        emotion_word_dict = {}
        for line in f:
            tmp_list = line.split()
            emotion_vec = tmp_list[1:]
            emotion_word_dict[tmp_list[0]] = list(map(int, emotion_vec))
    return emotion_word_dict

# テキストを入力し、トークンリストを返す
def get_token_list(tokenizer, text):
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True
    )
    encoding = { k: torch.tensor(v) for k, v in encoding.items() }
    return encoding

# # BERT_to_emotionのデータセットからリストを取得
# class BertToEmotionDataset(Dataset):
#     def __init__(self):
#         # txtデータの読み出し

class TokenListFileDataset(Dataset):
    def __init__(self, dirname, tokenizer):
        self.filenames = [os.path.join(dirname, n) for n in os.listdir(dirname)]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        encoding_list = []
        with open(self.filenames[idx], 'r') as f:
            for line in f:
                encoding_list.append(get_token_list(self.tokenizer, line))
        return encoding_list

if __name__ == '__main__':
    get_emotion_word_dict()

