SRC_FILE_NAME = '/workspace/emotion_analysis_test/nagasawa/data_src/wiki40b_with_emotion/wiki_40b_train_with_emotion.txt'
DST_FILE_NAME_HEAD = '/workspace/emotion_analysis_test/nagasawa/data_src/BERT_to_emotion/train/split/split_wiki_40b_train_with_emotion_'

def write_file(line_list, file_num):
	with open(DST_FILE_NAME_HEAD + '{:0>3}.txt'.format(file_num), 'w') as f:
		for line in line_list:
			f.write(line)

with open(SRC_FILE_NAME, 'r') as f:
	line_list = []
	count = 0
	file_num = 1
	for line in f:
		line_list.append(line)
		count += 1
		if count % 150000 == 0:
			print("write:{}".format(file_num))
			write_file(line_list, file_num)
			file_num += 1
			line_list = []
	print("write:{}".format(file_num))
	write_file(line_list, file_num)

