import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_module.tools import BertToEmoFileDataset
import tensorboard

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.bn = nn.BatchNorm1d(768)
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# 乱数シードを固定して再現性を確保
torch.manual_seed(0)

# インスタンス化
net = Net()

# 損失関数の選択
criterion = nn.CrossEntropyLoss()

# 最適化手法の選択
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# Trainデータセットの準備
# データセットが存在するディレクトリを指定
dataset_root_dir = "/workspace/dataset/data_src/BERT_to_emotion/only_emotion/train/split/"
train_file_dataset = BertToEmoFileDataset(dataset_root_dir)

# FileDataloader
train_file_loader = DataLoader(train_file_dataset)

# 乱数シード固定
torch.manual_seed(0)

# ネットワークのインスタンス化
net = Net()



