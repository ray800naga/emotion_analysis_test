# BERT_to_emoを１ファイルに付き１データとなるように分割
window_size = 0
mode = 'val'
SRC_FILE_NAME = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/BERT_to_emo_{1}.txt'.format(window_size, mode)
DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/split_one_line/{1}/split_BERT_to_emo_{1}_'.format(window_size, mode)

def write_file(line, file_num):
	with open(DST_FILE_NAME_HEAD + '{:0>8}.txt'.format(file_num), 'w') as f:
		f.write(line)

with open(SRC_FILE_NAME, 'r') as f:
	count = 0
	file_num = 1
	for line in f:
		count += 1
		print("write:{}".format(file_num))
		write_file(line, file_num)
		file_num += 1

