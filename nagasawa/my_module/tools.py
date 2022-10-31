from transformers import BertJapaneseTokenizer, BertModel

# bertでトークナイズされたトークン列のなかに感性語が含まれているか否かを確認
def has_emotion_word(tokens, emotion_word_list):
    for emo_w in emotion_word_list:
        for token in tokens:
            if emo_w == token:
                print(emo_w) # for debug
                return True
    return False

# 取得したバッチからデータセットを生成
def get_dataset_from_batch(batch, last_hidden_state):
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    emotion_word_dict = get_emotion_word_dict()
    batch_num = len(batch['input_ids'])
    print("batch_num: ", batch_num)
    for i in range(batch_num):
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
        emotion_word_idx = get_emotion_word_idx(tokens)

# tokensから感性語のインデックスのリストを返す。
def get_emotion_word_idx(tokens):
    idx_list = []
    emotion_word_dict = get_emotion_word_dict()
    for idx, token in enumerate(tokens):
        for emotion_word in emotion_word_dict.keys():
            if token == emotion_word:
                idx_list.append(idx)
                break
    print('idx_list:', idx_list)
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

if __name__ == '__main__':
    get_emotion_word_dict()