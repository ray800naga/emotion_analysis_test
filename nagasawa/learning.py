# %%
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_module.tools import BertToEmoFileDataset, BertToEmoDirectDataset, calc_loss, EarlyStopping
from tqdm import tqdm
import pickle
import datetime

# %%
# データセットの準備
# データセットが存在するディレクトリを指定

# window_sizeを設定
window_size = 3

#ファイル分割バージョン(省メモリ設計)	1ファイル１バッチ
dataset_root_dir = "/workspace/dataset/data_src/BERT_to_emotion/window_size_{}/split/train/".format(window_size)
train_file_dataset = BertToEmoFileDataset(dataset_root_dir)
dataset_root_dir = "/workspace/dataset/data_src/BERT_to_emotion/window_size_{}/split/val/".format(window_size)
val_file_dataset = BertToEmoFileDataset(dataset_root_dir)
dataset_root_dir = "/workspace/dataset/data_src/BERT_to_emotion/window_size_{}/split/test/".format(window_size)
test_file_dataset = BertToEmoFileDataset(dataset_root_dir)

# %%
# Pickle化したデータセットを読み込み
# with open("/workspace/dataset/data_src/BERT_to_emotion/windows_size_{}/dataset_list_window0.bin", 'rb') as p:
# 	dataset_list = pickle.load(p)
# train_dataset = dataset_list[0]
# val_dataset = dataset_list[1]
# test_dataset = dataset_list[2]

# %%
# ハイパーパラメータ
batch_size = 1
max_epoch = 10000

# 設定
num_workers = 12
date = str(datetime.datetime.today().date())
description = "batchnorm_400dim_MSE_window_3_bn_change"
model_path = "/workspace/dataset/data/model/{}_{}.pth".format(date, description)
print(model_path)

# %%
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.bn = nn.BatchNorm1d(400)
        self.fc1 = nn.Linear(768, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        x = torch.sigmoid(x)
        return x


# %%
# GPUの設定状況に基づいたデバイスの選択
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 乱数シードを固定して再現性を確保
torch.manual_seed(0)

# インスタンス化・デバイスへの転送
net = Net().to(device)

# 損失関数の選択
# criterion = nn.BCELoss()	# binary cross entropy
criterion = nn.MSELoss()	# mean square loss

# 最適化手法の選択
optimizer = torch.optim.Adam(net.parameters()) # default: lr=1e-3

# %%
# FileDataloader
train_file_loader = DataLoader(train_file_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_file_loader = DataLoader(val_file_dataset, batch_size=batch_size, num_workers=num_workers)
test_file_loader = DataLoader(test_file_dataset, batch_size=batch_size, num_workers=num_workers)

# %%
# early stopping
earlystopping  = EarlyStopping(patience=5, verbose=True, path=model_path)

# %%
# ネットワークの学習
for epoch in range(max_epoch):
    print("epoch:", epoch)
    loss_list = []
    for batch in tqdm(train_file_loader):
        x, t = batch
        x = x.view(-1, 768)
        t = t.view(-1, 10)
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        y = net(x)

        loss = criterion(y, t)

        loss.backward()

        loss_list.append(loss)

        optimizer.step()
        
    train_loss_avg = torch.tensor(loss_list).mean()
    print("val_loss calc...")
    val_loss_avg = calc_loss(net, val_file_loader, criterion, device)
    print("train_loss: {}, val_loss: {}".format(train_loss_avg, val_loss_avg))
    earlystopping(val_loss_avg, net) #callメソッド呼び出し
    if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
        print("Early Stopping!")
        break

# %%
best_net = Net()
best_net.load_state_dict(torch.load(model_path))
test_loss_avg = calc_loss(best_net, test_file_loader, criterion, device)
print("test_loss: {}".format(test_loss_avg))
print("done!")

# %%



