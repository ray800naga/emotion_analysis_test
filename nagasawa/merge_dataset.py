# 分割されたデータセットを1ファイルにまとめる
import os
from tqdm import tqdm

window_size = 0
mode = "test"

output_root_dirname = "/workspace/dataset/data_src/BERT_to_emotion/window_size_{}/{}/wakati/".format(window_size, mode)
output_file_name = "BERT_to_emo_{}.txt".format(mode)
input_root_dirname = output_root_dirname + "split/"

file_names = [os.path.join(input_root_dirname, n) for n in os.listdir(input_root_dirname)]

each_file_line_count = 0
with open(output_root_dirname + output_file_name, 'w') as f_w:
	for file_name in tqdm(file_names):
		with open(file_name, 'r') as f_r:
			for line in f_r:
				each_file_line_count += 1
				f_w.write(line)
print("done!")

# データ数をカウント
with open(output_root_dirname + output_file_name, 'r') as f:
	count = 0
	for line in f:
		count += 1
print("{}: {}".format(output_file_name, count))
print("each_file_line_count: ", each_file_line_count)