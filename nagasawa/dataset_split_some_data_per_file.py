# wiki40b_with_emotionのtrainファイルを１ファイルに付き150000データとなるように分割
# SRC_FILE_NAME = '/workspace/dataset/data_src/wiki40b_with_emotion/val/wakati/wiki_40b_val_with_emotion.txt'
# DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/wiki40b_with_emotion/val/wakati/split/split_wiki_40b_val_with_emotion_'

# BERT_to_emoのファイルを1ファイルにつき16384データとなるように分割
WINDOW_SIZE = 3
MODE = 'test'
SIZE =  16384
SRC_FILE_NAME = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/BERT_to_emo_{1}.txt'.format(WINDOW_SIZE, MODE)
DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/split/{1}/split_BERT_to_emo_{1}_'.format(WINDOW_SIZE, MODE)

def write_file(data_list, file_num):
	with open(DST_FILE_NAME_HEAD + '{:0>8}.txt'.format(file_num), 'w') as f:
		for line in data_list:
			f.write(line)

with open(SRC_FILE_NAME, 'r') as f:
	count = 0
	file_num = 1
	data_list = []
	for line in f:
		count += 1
		data_list.append(line)
		if count % SIZE == 0:
			print("write:{}".format(file_num))
			write_file(data_list, file_num)
			file_num += 1
			data_list = []
	print("write:{}".format(file_num))
	write_file(data_list, file_num)
	print("done!")
