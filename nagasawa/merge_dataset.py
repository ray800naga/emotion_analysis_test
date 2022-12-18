# 分割されたデータセットを1ファイルにまとめる
import os
from tqdm import tqdm
import sys
import slackweb
url = "https://hooks.slack.com/services/"
url = url + "T2AUFHDPT/B04D24YPQNS/oLeAqzdAfXiJAH4txODTD9ys"
slack = slackweb.Slack(url=url)

args = sys.argv

window_size = 3
mode = args[1]
min_output = 1
BERT = False

if BERT == True:
	input_root_dirname = "/workspace/dataset/data_src/BERT_to_emotion/window_size_{}/min_{}/split/{}".format(window_size, min_output, mode)
	output_file_name = "BERT_to_emo_{}.txt".format(mode)
	output_root_dirname = "/workspace/SSD/BERT_to_emotion/window_size_{}/min_{}/".format(window_size, min_output)
else:
	input_root_dirname = "/workspace/dataset/data_src/embed_to_emotion/window_size_{}/min_{}/split/{}".format(window_size, min_output, mode)
	output_file_name = "embed_to_emo_{}.txt".format(mode)
	output_root_dirname = "/workspace/SSD/embed_to_emotion/window_size_{}/min_{}/".format(window_size, min_output)

file_names = [os.path.join(input_root_dirname, n) for n in os.listdir(input_root_dirname)]

each_file_line_count = 0
with open(output_root_dirname + output_file_name, 'w') as f_w:
	for file_name in tqdm(file_names):
		with open(file_name, 'r') as f_r:
			for line in f_r:
				each_file_line_count += 1
				f_w.write(line)

# データ数をカウント
# with open(output_root_dirname + output_file_name, 'r') as f:
# 	count = 0
# 	for line in f:
# 		count += 1
# print("{}: {}".format(output_file_name, count))
# print("each_file_line_count: ", each_file_line_count)
print("done!")
slack.notify(text="merge dataset done : {}".format(output_root_dirname + output_file_name))