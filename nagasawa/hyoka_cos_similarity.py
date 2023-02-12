import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def put_list_into_file(list, method, word_group):
	with open(f"/home/student/docker_share/dataset/data/hyoka/jikken_1/{method}/cos_sim_{method}_{word_group}.txt", 'w') as f:
		for num in list:
			f.write(f"{num}\n")

method_list = ["BERT+weight", "BERT", "embedding"]
for method in method_list:
	with open("/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_all.txt", 'r') as f_human:
		with open(f"/home/student/docker_share/dataset/data/hyoka/jikken_1/{method}/output_emotion_vector_{method}.txt", 'r') as f_output:
			cos_sim_list_all = []
			cos_sim_list_normal = []
			cos_sim_list_kansei_tagi = []
			cos_sim_list_kansei = []
			for i in range(88):
				human_vector = list(map(float, f_human.readline()[:-1].split('\t')[2:]))
				output_vector = list(map(float, f_output.readline()[1:-2].split(', ')))
				human_vector = np.array(human_vector)
				output_vector = np.array(output_vector)
				cos_sim_list_all.append(cos_sim(human_vector, output_vector))
				if 0 <= i and i <= 35:
					cos_sim_list_normal.append(cos_sim(human_vector, output_vector))
				elif 36 <= i and i <= 69:
					cos_sim_list_kansei_tagi.append(cos_sim(human_vector, output_vector))
				elif 70 <= i and i <= 87:
					cos_sim_list_kansei.append(cos_sim(human_vector, output_vector))
			cos_sim_list_all = np.array(cos_sim_list_all)
			cos_sim_list_normal = np.array(cos_sim_list_normal)
			cos_sim_list_kansei = np.array(cos_sim_list_kansei)
			cos_sim_list_kansei_tagi = np.array(cos_sim_list_kansei_tagi)

			print(f"{method}-all: {np.mean(cos_sim_list_all)}")
			print(cos_sim_list_all)
			put_list_into_file(cos_sim_list_all, method, "all")
			print(f"{method}-noramal: {np.mean(cos_sim_list_normal)}")
			print(cos_sim_list_normal)
			put_list_into_file(cos_sim_list_normal, method, "normal")
			print(f"{method}-kansei: {np.mean(cos_sim_list_kansei)}")
			print(cos_sim_list_kansei)
			put_list_into_file(cos_sim_list_kansei, method, "kansei")
			print(f"{method}-kansei_tagi: {np.mean(cos_sim_list_kansei_tagi)}")
			print(cos_sim_list_kansei_tagi)
			put_list_into_file(cos_sim_list_kansei_tagi, method, "kansei_tagi")
