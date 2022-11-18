# %%
import torch
from transformers import BertJapaneseTokenizer, BertModel
import ipadic
from tqdm import tqdm

# %%
# 4-3
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

# %%
# text_listに格納
# with open('data_src/wiki40b_with_emotion/wiki_40b_train_with_emotion.txt', 'r') as f_read:
#     text_list = []
#     for line in f_read:
#         text_list.append(line)
# print(len(text_list))

# %%
import torch
from torch.utils.data import DataLoader
from my_module.tools import TokenListFileDataset
# 文章の符号化

text_file_dir = "/workspace/dataset/data_src/wiki40b_with_emotion/val/wakati/split/"
file_dataset = TokenListFileDataset(text_file_dir, tokenizer)

# dataset_for_loader = []
# count = 0 # for debug
# for text in tqdm(text_list):
#     encoding = tokenizer(
#         text,
#         max_length=128,
#         padding='max_length',
#         truncation=True
#     )
#     encoding = { k: torch.tensor(v) for k, v in encoding.items() }
#     dataset_for_loader.append(encoding)
#     # count += 1 # for debug
#     # if count == 100:
#     #     break

# %%
# FileDataLoader生成
file_loader = DataLoader(file_dataset, num_workers=2)

# %%
# DataLoader動作確認
# for idx, batch in enumerate(file_loader):
#     print("batch:", idx)
#     print(batch[0]["input_ids"].size())

# %%
bert = BertModel.from_pretrained(model_name)

bert = bert.cuda()

# %%
from my_module.tools import get_dataset_from_batch
from concurrent.futures import ProcessPoolExecutor
count = 0
with torch.no_grad():
    # データセット出力先を指定
    output_name_head = '/workspace/dataset/data_src/BERT_to_emotion/only_emotion/val/wakati/split/BERT_to_emo_val_'
    file_count = 1
    with ProcessPoolExecutor(max_workers=2) as executor:
        for file in file_loader:
            print("file_count: {} / {}".format(file_count, file_loader.__len__()))
            batch_loader = DataLoader(file, batch_size=128)
            batch_count = 1
            for batch in tqdm(batch_loader):
                # データをGPUに乗せる
                batch = {k: v.squeeze().cuda() for k, v in batch.items()}
                # BERTでの処理
                output = bert(**batch)
                last_hidden_state = output.last_hidden_state
                last_hidden_state = last_hidden_state.cpu().numpy().tolist()
                batch = {k: v.cpu().numpy().tolist() for k, v in batch.items()}
                # executor.submit(get_dataset_from_batch, batch, last_hidden_state, file_count, batch_count, output_name_head)
                get_dataset_from_batch(batch, last_hidden_state, file_count, batch_count, output_name_head)
                batch_count += 1
            file_count += 1
print('done!')

# %%
# from my_module.tools import get_dataset_from_batch
# from concurrent.futures import ProcessPoolExecutor
# count = 0
# with torch.no_grad():
#     # データセット出力先を指定
#     output_name_head = '/workspace/dataset/data_src/BERT_to_emotion/only_emotion/val/wakati/BERT_to_emo_train_'
#     file_count = 1
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         for file in file_loader:
#             print("file_count: {} / {}".format(file_count, file_loader.__len__()))
#             batch_loader = DataLoader(file, batch_size=256)
#             batch_count = 1
#             for batch in tqdm(batch_loader):
#                 # データをGPUに乗せる
#                 encoding_list, text_list = batch
#                 encoding_list = {k: v.squeeze().cuda() for k, v in encoding_list.items()}
#                 # BERTでの処理
#                 output = bert(**encoding_list)
#                 last_hidden_state = output.last_hidden_state
#                 last_hidden_state = last_hidden_state.cpu().numpy().tolist()
#                 encoding_list = {k: v.cpu().numpy().tolist() for k, v in encoding_list.items()}
#                 executor.submit(get_dataset_from_batch, encoding_list, text_list, last_hidden_state, file_count, batch_count, output_name_head)
#                 batch_count += 1
#             file_count += 1
# print('done!')

# %%



