from transformers import BertJapaneseTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch
import os
import statistics
from tqdm import tqdm
import numpy as np
from MeCab import Tagger
import ipadic
from concurrent.futures import ProcessPoolExecutor

NON_STOP_WORD = ['動詞', '名詞', '形容詞', '形容動詞']
STOP_JOUKEN = ['代名詞', '接尾', '非自立', '数']
STOP_WORD = ['する', 'いる', 'ある', 'なる', 'ない', 'できる', 'おり', '行う', 'ば', 'ら']

# MeCabで形態素解析され、原型に戻されたリストのなかに感性語が含まれているか否かを確認
def has_emotion_word(genkei_list, emotion_word_list):
    for emo_w in emotion_word_list:
        for genkei in genkei_list:
            if emo_w == genkei:
                # print(emo_w) # for debug
                return True
    return False

# textから原型のリストを取得
def get_genkei_list(text, tagger):
    tagger_list = tagger.parse(text).split("\n")
    del tagger_list[-2:]
    genkei_list = []
    # print(line) # for debug
    for i in tagger_list:
        i = i.split("\t")
        i += i[1].split(",")
        del i[1]
        genkei_list.append(i[7])
    return genkei_list

# 取得したバッチからデータセットを生成
def get_dataset_from_batch(encoding_list, last_hidden_state, file_count, batch_count, output_name_head, window_size):
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    tagger = Tagger(ipadic.MECAB_ARGS)
    emotion_word_dict = get_emotion_word_dict()
    batch_len = len(encoding_list['input_ids'])
    # print("batch_num: ", batch_num)
    with open(output_name_head + "{:0>8}_{:0>8}.txt".format(file_count, batch_count), 'w') as f:
        for i in range(batch_len):
            # input_ids には1文のid列が格納されている。
            tokens = tokenizer.convert_ids_to_tokens(encoding_list['input_ids'][i])
            # taggers = tagger.parse(text).split("\n")
            del tokens[0]   # [CLS]を削除
            del last_hidden_state[i][0]    # [CLS]に対応する出力も削除
            sep_idx = tokens.index('[SEP]') # [SEP]のindexを取得
            del tokens[sep_idx:]    # [SEP]以降を削除
            del last_hidden_state[i][sep_idx:] # 対応する出力も削除
            # tokenとbertからの出力で次元数に違いがあればエラーとし、処理をとばす。
            if len(tokens) != len(last_hidden_state[i]):
                print("error")
                continue
            # print(tokens) # for debug
            # サブワード化されたtokenを集約し、単語単位に変換
            token_idx_list_with_no_subword = concat_subwords(tokens)
            # print("no_subword: ", token_idx_list_with_no_subword) # for debug
            # ウィンドウカウント対象単語に絞り込み
            tokens_idx_list_for_window = get_list_for_window_size_consideration(tokens, token_idx_list_with_no_subword, tagger)
            # ウィンドウカウント対象単語を表示
            # show_tokens_idx_list_for_window(tokens, tokens_idx_list_for_window) # for debug
            # print("window_idx_list", tokens_idx_list_for_window)
            # 感性語のtoken indexを取得 返り値はtokens_idx_list_for_windowでのindex
            emotion_word_idx_list = get_emotion_word_idx(tokens, tokens_idx_list_for_window, tagger)
            # 各感性語について、データセットを生成
            for emotion_word_idx in emotion_word_idx_list:
                emotion_word = get_word_from_subwords(tokens, tokens_idx_list_for_window[emotion_word_idx])
                emotion_word_genkei = get_genkei(emotion_word, tagger)
                emotion_vector = emotion_word_dict[emotion_word_genkei]
                for j in tokens_idx_list_for_window[emotion_word_idx]:
                    # print("{} : {}".format(tokens[j], emotion_vector)) # for debug
                    # 感性語には距離情報(0)を最後のフィールドに付加
                    f.write("{}\t{}\t0\n".format(last_hidden_state[i][j], emotion_vector))
                # ウィンドウサイズ分の周辺単語も、emotion_vectorを持つ単語としてデータセットに登録
                for window in range(1, window_size + 1):
                    # 前方向
                    outer_idx = emotion_word_idx - window
                    if outer_idx >= 0:
                        for k in tokens_idx_list_for_window[outer_idx]:
                            # print("{} : {}".format(tokens[k], weighted_emotion_vector)) # for debug
                            f.write("{}\t{}\t{}\n".format(last_hidden_state[i][k], emotion_vector, window))
                    # 後方向
                    outer_idx = emotion_word_idx + window
                    if outer_idx < len(tokens_idx_list_for_window):
                        for k in tokens_idx_list_for_window[outer_idx]:
                            # print("{} : {}".format(tokens[k], weighted_emotion_vector)) # for debug
                            f.write("{}\t{}\t{}\n".format(last_hidden_state[i][k], emotion_vector, window))

# 取得したバッチからデータセットを生成(並列化)
def get_dataset_from_batch_multi_process(encoding_list, last_hidden_state, file_count, batch_count, output_name_head, window_size):
    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    tagger = Tagger(ipadic.MECAB_ARGS)
    emotion_word_dict = get_emotion_word_dict()
    batch_len = len(encoding_list['input_ids'])
    # print("batch_num: ", batch_num)
    with open(output_name_head + "{:0>8}_{:0>8}.txt".format(file_count, batch_count), 'w') as f:
        with ProcessPoolExecutor(max_worker=10) as executor:
            future_list = []
            for i in range(batch_len):
                future_list.append(executor.submit(get_dataset_from_batch_multi_process_child, tokenizer, encoding_list['input_ids'][i], last_hidden_state[i], tagger, emotion_word_dict, window_size))
        for future in future_list:
            result = future.result()
            if result == False:
                continue
            else:
                for line in result:
                    f.write(line)
            
def get_dataset_from_batch_multi_process_child(tokenizer, encoding_list_input_ids_i, last_hidden_state_i, tagger, emotion_word_dict, window_size):
    # input_ids には1文のid列が格納されている。
    tokens = tokenizer.convert_ids_to_tokens(encoding_list_input_ids_i)
    # taggers = tagger.parse(text).split("\n")
    del tokens[0]   # [CLS]を削除
    del last_hidden_state_i[0]    # [CLS]に対応する出力も削除
    sep_idx = tokens.index('[SEP]') # [SEP]のindexを取得
    del tokens[sep_idx:]    # [SEP]以降を削除
    del last_hidden_state_i[sep_idx:] # 対応する出力も削除
    # tokenとbertからの出力で次元数に違いがあればエラーとし、処理をとばす。
    if len(tokens) != len(last_hidden_state_i):
        print("error")
        return False
    # print(tokens) # for debug
    # サブワード化されたtokenを集約し、単語単位に変換
    token_idx_list_with_no_subword = concat_subwords(tokens)
    # print("no_subword: ", token_idx_list_with_no_subword) # for debug
    # ウィンドウカウント対象単語に絞り込み
    tokens_idx_list_for_window = get_list_for_window_size_consideration(tokens, token_idx_list_with_no_subword, tagger)
    # ウィンドウカウント対象単語を表示
    # show_tokens_idx_list_for_window(tokens, tokens_idx_list_for_window) # for debug
    # print("window_idx_list", tokens_idx_list_for_window)
    # 感性語のtoken indexを取得 返り値はtokens_idx_list_for_windowでのindex
    emotion_word_idx_list = get_emotion_word_idx(tokens, tokens_idx_list_for_window, tagger)
    # 各感性語について、データセットを生成
    output_list = []
    for emotion_word_idx in emotion_word_idx_list:
        emotion_word = get_word_from_subwords(tokens, tokens_idx_list_for_window[emotion_word_idx])
        emotion_word_genkei = get_genkei(emotion_word, tagger)
        emotion_vector = emotion_word_dict[emotion_word_genkei]
        for j in tokens_idx_list_for_window[emotion_word_idx]:
            # print("{} : {}".format(tokens[j], emotion_vector)) # for debug
            # 感性語には距離情報(0)を最後のフィールドに付加
            output_list.append("{}\t{}\t0\n".format(last_hidden_state_i[j], emotion_vector))
        # ウィンドウサイズ分の周辺単語も、emotion_vectorを持つ単語としてデータセットに登録
        for window in range(1, window_size + 1):
            # 前方向
            outer_idx = emotion_word_idx - window
            if outer_idx >= 0:
                for k in tokens_idx_list_for_window[outer_idx]:
                    # print("{} : {}".format(tokens[k], weighted_emotion_vector)) # for debug
                    output_list.append("{}\t{}\t{}\n".format(last_hidden_state_i[k], emotion_vector, window))
            # 後方向
            outer_idx = emotion_word_idx + window
            if outer_idx < len(tokens_idx_list_for_window):
                for k in tokens_idx_list_for_window[outer_idx]:
                    # print("{} : {}".format(tokens[k], weighted_emotion_vector)) # for debug
                    output_list.append("{}\t{}\t{}\n".format(last_hidden_state_i[k], emotion_vector, window))
    return output_list
    
def show_tokens_idx_list_for_window(tokens, tokens_idx_list_for_window):
    for idx_list in tokens_idx_list_for_window:
        for idx in idx_list:
            print(tokens[idx], end=' ')
        print('/', end=' ')
    print()

# tokensから感性語のインデックスのリストを返す。
def get_emotion_word_idx(tokens, tokens_dix_list_for_window, tagger):
    emo_token_idx_list = []
    emotion_word_dict = get_emotion_word_dict()
    for outer_idx, token_idx_list in enumerate(tokens_dix_list_for_window):
        word = get_word_from_subwords(tokens, token_idx_list)
        genkei = get_genkei(word, tagger)
        for emotion_word in emotion_word_dict.keys():
            if genkei == emotion_word:
                emo_token_idx_list.append(outer_idx)
                break
    # print('emo_token_idx_list:', emo_token_idx_list)
    return emo_token_idx_list

# 単語の原型を取得
def get_genkei(word, tagger):
    genkei = tagger.parse(word).split('\n')[0].split('\t')[1].split(',')[6]
    return genkei

# サブワードを連結したidリストを元に、##を取り除いて連結した単語の文字列を取得
def get_word_from_subwords(tokens, token_idx_list):
    word = tokens[token_idx_list[0]]
    for i in range(1, len(token_idx_list)):
        word = word + tokens[token_idx_list[i]][2:]
    return word



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

# tokensのサブワード化を解消
def concat_subwords(tokens):
    tokens_idx_list_no_subword = []
    tokens_in_one_tagger = []
    token_idx = 0
    # print(len(tokens)) # for debug
    while token_idx < len(tokens):
        tokens_in_one_tagger.append(token_idx)
        # print("added_token_idx(main): ", token_idx)
        token_idx += 1
        if token_idx == len(tokens):
            tokens_idx_list_no_subword.append(tokens_in_one_tagger)
            break
        while "##" in tokens[token_idx]:
            tokens_in_one_tagger.append(token_idx)
            # print("added_token_idx(sub): ", token_idx)
            token_idx += 1
            if token_idx == len(tokens):
                break
        tokens_idx_list_no_subword.append(tokens_in_one_tagger)
        tokens_in_one_tagger = []
    return tokens_idx_list_no_subword

# サブワードを連結したtokenリストの中から、考慮対象の品詞のものだけを抽出
def get_list_for_window_size_consideration(tokens, tokens_idx_list_no_subword, tagger):
    tokens_idx_list_for_window_consideration = []
    for token_idxs in tokens_idx_list_no_subword:
        word = get_word_from_subwords(tokens, token_idxs)
        # print(word) # for debug
        # 1単語に対し、分かち書きで複数に別れた場合は除外
        if len(tagger.parse(word).split('\n')) != 3:
            continue
        tagger_list = tagger.parse(word).split('\n')[0].split('\t')[1].split(',')
        hinshi = tagger_list[0]
        jouken = tagger_list[1]
        genkei = tagger_list[6]
        if (hinshi in NON_STOP_WORD) and (jouken not in STOP_JOUKEN) and (genkei not in STOP_WORD):
            tokens_idx_list_for_window_consideration.append(token_idxs)
    return tokens_idx_list_for_window_consideration
        
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
        input_vec_list = []
        output_vec_list = []
        with open(self.filenames[idx], 'r') as f:
            for line in f:
                # 最後の改行文字を落として、タブ文字で分割
                data = line[:-1].split('\t')
                # 入力値の処理(768dim)
                # 大かっこを落として、「, 」で分割
                input_vec = data[0][1:-1].split(', ')
                # floatに変換
                input_vec = list(map(float, input_vec))
                # 出力地の処理(10dim)
                # 大かっこを落として、「, 」で分割
                output_vec = data[1][1:-1].split(', ')
                # floatに変換
                output_vec = list(map(float, output_vec))
                # リストに格納
                input_vec_list.append(input_vec)
                output_vec_list.append(output_vec)
        input_vec_list = torch.tensor(input_vec_list)
        output_vec_list = torch.tensor(output_vec_list)
        return input_vec_list, output_vec_list

# BERT特徴量と感情ベクトルのデータセットを取得(ファイルまるごと一気に読み込み)
class BertToEmoDirectDataset(Dataset):
    def __init__(self, dirname):
        with open(dirname, 'r') as f:
            input_vec_list = []
            output_vec_list = []
            for line in f:
                data = line[:-1].split('\t')
                input_vec = data[0][1:-1].split(', ')
                input_vec = list(map(float, input_vec))
                output_vec = data[1][1:-1].split(', ')
                output_vec = list(map(float, output_vec))
                input_vec = torch.tensor(input_vec)
                input_vec_list.append(input_vec)
                output_vec = torch.tensor(output_vec)
                output_vec_list.append(output_vec)
        self.input_vec_list = input_vec_list
        self.output_vec_list = output_vec_list

    def __len__(self):
        return len(self.input_vec_list)

    def __getitem__(self, idx):
        return self.input_vec_list[idx], self.output_vec_list[idx]



# lossの平均値を算出(val, testのloss計算用)
def calc_loss(net, file_loader, criterion, device):
    with torch.no_grad():
        loss_list = []
        for batch in tqdm(file_loader):
            x, t = batch
            x = x.squeeze().to(device)
            t = t.squeeze().to(device)
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

