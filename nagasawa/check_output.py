from transformers import BertJapaneseTokenizer, BertModel
import ipadic
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_module.tools import get_token_list
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.bn = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        x = torch.sigmoid(x)
        return x

model_weight_path = "/workspace/dataset/data/model/2022-11-26_batchnorm_400dim_BCE_window_3_bn_change.pth"
net = Net()
net.load_state_dict(torch.load(model_weight_path))

# BERTモデルの指定
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

# GPUの設定状況に基づいたデバイスの選択
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

bert = bert.to(device)
net = net.to(device)

# input_sentence = input("文章を入力：")
input_sentence = "大切なものをなくしてしまい、悲しい。"
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

    print(tokens)

    last_hidden_state = torch.tensor(last_hidden_state)
    last_hidden_state = last_hidden_state.to(device)

    output = net(last_hidden_state)
    output = output.cpu().numpy()
    for idx, token in enumerate(tokens):
        print(token)
        print(output[idx])
        plt.bar(left, output[idx], tick_label=emotion_label)
        plt.ylim(0, 1)
        plt.title(token)
        plt.savefig("./{}.png".format(idx))
        plt.clf()