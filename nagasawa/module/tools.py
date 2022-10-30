from transformers import BertJapaneseTokenizer, BertModel

# bertでトークナイズされたトークン列のなかに感性語が含まれているか否かを確認
def has_emotion_word(tokens, emotion_word_list):
    for emo_w in emotion_word_list:
        for token in tokens:
            if emo_w == token:
                print(emo_w) # for debug
                return True
    return False

