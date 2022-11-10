from my_module.tools import BertToEmoDirectDataset
import pickle

print("loading train dataset")
dataset_dir = "/workspace/dataset/data_src/BERT_to_emotion/only_emotion/train/BERT_to_emo_train.txt"
train_dataset = BertToEmoDirectDataset(dataset_dir)

print("loading val dataset")
dataset_dir = "/workspace/dataset/data_src/BERT_to_emotion/only_emotion/val/BERT_to_emo_val.txt"
val_dataset = BertToEmoDirectDataset(dataset_dir)

print("loading test dataset")
dataset_dir = "/workspace/dataset/data_src/BERT_to_emotion/only_emotion/test/BERT_to_emo_test.txt"
test_dataset = BertToEmoDirectDataset(dataset_dir)

dataset_list = [train_dataset, val_dataset, test_dataset]

print("writing...")
with open('dataset_list_window0.bin', 'wb') as f:
	pickle.dump(dataset_list, f)