from transformers import BertJapaneseTokenizer, BertModel
import ipadic
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_module.tools import get_token_list, concat_subwords, get_list_for_window_size_consideration, show_tokens_idx_list_for_window, get_word_from_subwords
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from MeCab import Tagger
import ipadic

def is_in_list_for_window(idx, tokens_idx_list_for_window):
    for token_idx_list in tokens_idx_list_for_window:
        if idx in token_idx_list:
            return True
    return False

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.bn = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn(x)
        x = F.relu(x)
        # x = F.leaky_relu(x)
        # x = F.mish(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        x = torch.sigmoid(x)
        return x

model_weight_path = "/workspace/dataset/data/model/False_512_400dim_MSE_window_3_relu.pth"
net = Net()
net.load_state_dict(torch.load(model_weight_path))

# BERTモデルの指定
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name, output_hidden_states=True)

# MeCabの設定
tagger = Tagger(ipadic.MECAB_ARGS)

# GPUの設定状況に基づいたデバイスの選択
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("device:", device)

bert = bert.to(device)
net = net.to(device)

# input_sentence = input("文章を入力：")
# input_sentence = "大切なものをなくしてしまい、悲しい。"
input_sentence = "今日は海へドライブに行って、とても気持ち良かった。"
# input_sentence = "今日は、ドライブ中に交通事故に巻き込まれてしまった。"
# input_sentence = "道の真ん中で転んでしまい恥ずかしかったので、走ってその場を立ち去った。"
encoding = get_token_list(tokenizer, input_sentence)
encoding = {k: v.unsqueeze(dim=0).to(device) for k, v in encoding.items()}

# 出力グラフ設定
left = np.array([i for i in range(1, 11)])
height = ([0.2 * i for i in range(1, 6)])
emotion_label = ["喜", "怒", "哀", "怖", "恥", "好", "厭", "昴", "安", "驚"]

with torch.no_grad():
    output = bert(**encoding)
    print(output.keys())
    embeddings = output.hidden_states[0]
    embeddings = embeddings.squeeze().cpu().numpy().tolist()

    encoding = {k: v.cpu().numpy().tolist() for k, v in encoding.items()}

    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

    del tokens[0]   # [CLS]を削除
    del embeddings[0]    # [CLS]に対応する出力も削除
    sep_idx = tokens.index('[SEP]') # [SEP]のindexを取得
    del tokens[sep_idx:]    # [SEP]以降を削除
    del embeddings[sep_idx:] # 対応する出力も削除

    print("入力トークン列")
    print(tokens)
    token_idx_list_with_no_subword = concat_subwords(tokens)
    print("サブワード連結後")
    show_tokens_idx_list_for_window(tokens, token_idx_list_with_no_subword)
    tokens_idx_list_for_window = get_list_for_window_size_consideration(tokens, token_idx_list_with_no_subword, tagger)
    print("ウィンドウカウント対象トークン列")
    show_tokens_idx_list_for_window(tokens, tokens_idx_list_for_window)
    print(tokens_idx_list_for_window)
    
    embeddings = torch.tensor(embeddings)
    embeddings = embeddings.to(device)

    output = net(embeddings)
    output = output.cpu().numpy()

    count = 0
    for token_idx_list in tokens_idx_list_for_window:
        count += 1
        word = get_word_from_subwords(tokens, token_idx_list)
        output_list = []
        for token_idx in token_idx_list:
            output_list.append(output[token_idx])
        output_average = np.average(output_list, axis=0)
        print(word)
        print(output_average)
        plt.bar(left, output_average, tick_label=emotion_label)
        plt.ylim(0, 1)
        plt.title(word)
        plt.savefig("/workspace/{:0=3}_{}.png".format(count, word))
        plt.clf()

    