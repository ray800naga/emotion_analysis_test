from transformers import BertJapaneseTokenizer, BertModel
import ipadic
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_module.tools import get_token_list

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.bn = nn.BatchNorm1d(768)
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 10)
        # self.fc2 = nn.Linear(400, 100)
        # self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # x = F.relu(x)
        # x = self.fc3(x)
        return x

model_weight_path = "./checkpoint_model.pth"
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
input_sentence = "いい天気ですね。"
encoding = get_token_list(tokenizer, input_sentence)
encoding = {k: v.unsqueeze(dim=0).to(device) for k, v in encoding.items()}

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

	last_hidden_state = torch.tensor(last_hidden_state)
	last_hidden_state = last_hidden_state.to(device)

	output = net(last_hidden_state)
	print(output)