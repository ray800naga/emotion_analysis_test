import slackweb
url = "https://hooks.slack.com/services/"
url = url + "T2AUFHDPT/B04D24YPQNS/oLeAqzdAfXiJAH4txODTD9ys"
slack = slackweb.Slack(url=url)

import sys

args = sys.argv

# BERT_to_emoのファイルを1ファイルにつき512データとなるように分割
WINDOW_SIZE = 3
MODE = args[1]
SIZE =  512
MIN_OUTPUT = 1
BERT = False

if BERT == True:
	SRC_FILE_NAME = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/min_{1}/BERT_to_emo_{2}.txt'.format(WINDOW_SIZE, MIN_OUTPUT, MODE)
	DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/min_{1}/split_{2}/{3}/split_BERT_to_emo_{3}_'.format(WINDOW_SIZE, MIN_OUTPUT, SIZE, MODE)
else:
	SRC_FILE_NAME = '/workspace/dataset/data_src/embed_to_emotion/window_size_{0}/min_{1}/embed_to_emo_{2}.txt'.format(WINDOW_SIZE, MIN_OUTPUT, MODE)
	DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/embed_to_emotion/window_size_{0}/min_{1}/split_{2}/{3}/split_embed_to_emo_{3}_'.format(WINDOW_SIZE, MIN_OUTPUT, SIZE, MODE)

# wiki40b_with_emotionのtrainファイルを１ファイルに付き15000データとなるように分割
# MODE = "train"
# SIZE = 15000
# SRC_FILE_NAME = '/workspace/dataset/data_src/wiki40b_with_emotion/{0}/wakati/wiki_40b_{0}_with_emotion.txt'.format(MODE)
# DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/wiki40b_with_emotion/{0}/wakati/split/split_wiki_40b_{0}_with_emotion_'.format(MODE)


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
	slack.notify(text="split done! : {}".format(DST_FILE_NAME_HEAD))
