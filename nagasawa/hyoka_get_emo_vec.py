# 実験1
import glob

normal_words = ['仕事', '学校', 'もたらす', '試作', '軽い', 'リンゴ', '夏休み', '真似', '手順', '呼び寄せる', '果物', '石鹸', '海老', '最年長', 'エアコン', '地下道', '建て直', '積み重な']
kansei_tagi_words = ['気持ち', '涙', '思い']
kansei_words = ['驚か', '笑み', '切な', '恋し', '好む', '不満', '大嫌い', 'めでたい', '侮', '立腹', '安らか', '感慨', '心苦し', '名残惜し', '嬉し涙', '祝す', '叱りつける', '晴れ渡']
output_vector_path = '/home/student/docker_share/dataset/data/output/jikken/'
hyoka_path = '/home/student/docker_share/dataset/data/hyoka/jikken_1/'
situation_list = ['BERT', 'BERT+weight', 'embedding']

def get_and_write_data(situation, q_num, word):
	path = output_vector_path + "{}/".format(situation) + "Q{:0>2}/".format(q_num)
	file = glob.glob(path + "*{}*.txt".format(word))
	print(file)
	with open(file[0], 'r') as f:
		line = f.readline()
		with open(hyoka_path + f"{situation}/output_emotion_vector_{situation}.txt", 'a') as f_w:
			f_w.write(line + '\n')

for situation in situation_list:
	for q_num in range(1, 88 + 1):
		if 1 <= q_num and q_num <= 36:
			word = normal_words[(q_num - 1) // 2]
			get_and_write_data(situation, q_num, word)
		elif 37 <= q_num and q_num <= 70:
			if 37 <= q_num and q_num <= 48:
				word = kansei_tagi_words[0]
			elif 49 <= q_num and q_num <= 60:
				word = kansei_tagi_words[1]
			else:
				word = kansei_tagi_words[2]
			get_and_write_data(situation, q_num, word)
		elif 71 <= q_num and q_num <= 88:
			word = kansei_words[q_num - 71]
			get_and_write_data(situation, q_num, word)

