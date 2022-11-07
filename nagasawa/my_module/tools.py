from transformers import BertJapaneseTokenizer, BertModel
from torch.utils.data import Dataset
import torch
import os
import statistics
from tqdm import tqdm
import numpy as np

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
    with open('/workspace/dataset/data_src/w_senti_vocab.txt', 'r') as f:
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

# 文字列をトークン化したデータセットを取得（並列化のためにファイル分割へ対応）
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

# BERT特徴量と感情ベクトルのデータセットを取得(多量データ対応のため、ファイル分割対応→1ファイル1文)
class BertToEmoFileDataset(Dataset):
    def __init__(self, root_dirname):
        self.filenames = [os.path.join(root_dirname, n) for n in os.listdir(root_dirname)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with open(self.filenames[idx], 'r') as f:
            line = f.readline()
        data = line[:-1].split('\t')
        input_vec = data[0][1:-1].split(', ')
        input_vec = list(map(float, input_vec))
        output_vec = data[1][1:-1].split(', ')
        output_vec = list(map(float, output_vec))
        input_vec = torch.tensor(input_vec)
        output_vec = torch.tensor(output_vec)
        return input_vec, output_vec

# lossの平均値を算出(val, testのloss計算用)
def calc_loss(net, data_loader, criterion, device):
    with torch.no_grad():
        loss_list = []
        for batch in tqdm(data_loader):
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = net(x)

            loss = criterion(y, t)
            loss_list.append(loss)
        avg_loss = torch.tensor(loss_list).mean()
        return avg_loss

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

if __name__ == '__main__':
    print(get_emotion_word_dict())

