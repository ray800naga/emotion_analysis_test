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
import os

def is_in_list_for_window(idx, tokens_idx_list_for_window):
    for token_idx_list in tokens_idx_list_for_window:
        if idx in token_idx_list:
            return True
    return False

def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i,y_list[i-1],"{:.3f}".format(y_list[i-1]), ha="center")

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

model_weight_path = "/workspace/dataset/data/model/512_400dim_MSE_window_3_weight_relu.pth"
net = Net()
net.load_state_dict(torch.load(model_weight_path))

# BERTモデルの指定
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

# MeCabの設定
tagger = Tagger(ipadic.MECAB_ARGS)

# GPUの設定状況に基づいたデバイスの選択
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print("device:", device)

bert = bert.to(device)
net = net.to(device)

with open("/workspace/dataset/data_src/input_text.txt", 'r') as f:
    line_count = 1
    for input_sentence in f:
        os.makedirs("/workspace/dataset/data/output/{}/Q{:0=2}".format(model_weight_path.split('/')[-1], line_count), exist_ok=True)
        path = "/workspace/dataset/data/output/{}/Q{:0=2}".format(model_weight_path.split('/')[-1], line_count)
        
        encoding = get_token_list(tokenizer, input_sentence)
        encoding = {k: v.unsqueeze(dim=0).to(device) for k, v in encoding.items()}

        # 出力グラフ設定
        left = np.array([i for i in range(1, 11)])
        height = ([0.2 * i for i in range(1, 6)])
        emotion_label = ["喜", "怒", "哀", "怖", "恥", "好", "厭", "昴", "安", "驚"]

        with torch.no_grad():
            output = bert(**encoding)
            last_hidden_state = output.last_hidden_state
            last_hidden_state = last_hidden_state.squeeze().cpu().numpy().tolist()

            encoding = {k: v.cpu().numpy().tolist() for k, v in encoding.items()}

            tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

            del tokens[0]   # [CLS]を削除
            del last_hidden_state[0]    # [CLS]に対応する出力も削除
            sep_idx = tokens.index('[SEP]') # [SEP]のindexを取得
            del tokens[sep_idx:]    # [SEP]以降を削除
            del last_hidden_state[sep_idx:] # 対応する出力も削除

            print("入力トークン列")
            print(tokens)
            token_idx_list_with_no_subword = concat_subwords(tokens)
            print("サブワード連結後")
            show_tokens_idx_list_for_window(tokens, token_idx_list_with_no_subword)
            tokens_idx_list_for_window = get_list_for_window_size_consideration(tokens, token_idx_list_with_no_subword, tagger)
            print("ウィンドウカウント対象トークン列")
            show_tokens_idx_list_for_window(tokens, tokens_idx_list_for_window)
            print(tokens_idx_list_for_window)
            
            last_hidden_state = torch.tensor(last_hidden_state)
            last_hidden_state = last_hidden_state.to(device)

            output = net(last_hidden_state)
            output = output.cpu().numpy()

            count = 1
            for token_idx_list in tokens_idx_list_for_window:
                word = get_word_from_subwords(tokens, token_idx_list)
                output_list = []
                for token_idx in token_idx_list:
                    output_list.append(output[token_idx])
                output_average = np.average(output_list, axis=0)
                print(word)
                print(output_average)
                plt.bar(left, output_average, tick_label=emotion_label)
                plt.ylim(0, 1)
                add_value_label(left, output_average)
                plt.title(word)
                plt.savefig("{}/{:0=3}_{}.png".format(path, count, word))
                with open("{}/{:0=3}_{}.txt".format(path, count, word), 'w') as f_write:
                    f_write.write(str(output_average.tolist()))
                plt.clf()
                count += 1
        line_count += 1

