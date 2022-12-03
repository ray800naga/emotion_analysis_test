import slackweb
url = "https://hooks.slack.com/services/"
url = url + "T2AUFHDPT/B04D24YPQNS/oLeAqzdAfXiJAH4txODTD9ys"
slack = slackweb.Slack(url=url)

# BERT_to_emoのファイルを1ファイルにつき4096データとなるように分割
# WINDOW_SIZE = 3
# MODE = 'train'
# SIZE =  128
# SRC_FILE_NAME = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/BERT_to_emo_{1}.txt'.format(WINDOW_SIZE, MODE)
# DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/BERT_to_emotion/window_size_{0}/split_{2}/{1}/split_BERT_to_emo_{1}_'.format(WINDOW_SIZE, MODE, SIZE)

# wiki40b_with_emotionのtrainファイルを１ファイルに付き15000データとなるように分割
MODE = "train"
SIZE = 15000
SRC_FILE_NAME = '/workspace/dataset/data_src/wiki40b_with_emotion/{0}/wakati/wiki_40b_{0}_with_emotion.txt'.format(MODE)
DST_FILE_NAME_HEAD = '/workspace/dataset/data_src/wiki40b_with_emotion/{0}/wakati/split/split_wiki_40b_{0}_with_emotion_'.format(MODE)


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
