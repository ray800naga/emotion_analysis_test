# wiki40b_with_emotionのtrainファイルを１ファイルに付き150000データとなるように分割
SRC_FILE_NAME = '/workspace/dataset/data_src/wiki40b_with_emotion/val/wakati/wiki_40b_val_with_emotion.txt'
DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/wiki40b_with_emotion/val/wakati/split/split_wiki_40b_val_with_emotion_'

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
		if count % 15000 == 0:
			print("write:{}".format(file_num))
			write_file(data_list, file_num)
			file_num += 1
			data_list = []
	print("write:{}".format(file_num))
	write_file(data_list, file_num)
	print("done!")
