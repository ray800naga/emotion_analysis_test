import numpy as np

def euclid(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def put_list_into_file(list, method, word_group):
	with open(f"/home/student/docker_share/dataset/data/hyoka/jikken_1/{method}/euclid_{method}_{word_group}.txt", 'w') as f:
		for num in list:
			f.write(f"{num}\n")

method_list = ["BERT+weight", "BERT", "embedding"]
for method in method_list:
	with open("/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_all.txt", 'r') as f_human:
		with open(f"/home/student/docker_share/dataset/data/hyoka/jikken_1/{method}/output_emotion_vector_{method}.txt", 'r') as f_output:
			euclid_list_all = []
			euclid_list_normal = []
			euclid_list_kansei_tagi = []
			euclid_list_kansei = []
			for i in range(88):
				human_vector = list(map(float, f_human.readline()[:-1].split('\t')[2:]))
				output_vector = list(map(float, f_output.readline()[1:-2].split(', ')))
				human_vector = np.array(human_vector)
				output_vector = np.array(output_vector)
				# normalization for output_vector
				output_vector = output_vector / np.sum(output_vector)
				euclid_list_all.append(euclid(human_vector, output_vector))
				if 0 <= i and i <= 35:
					euclid_list_normal.append(euclid(human_vector, output_vector))
				elif 36 <= i and i <= 69:
					euclid_list_kansei_tagi.append(euclid(human_vector, output_vector))
				elif 70 <= i and i <= 87:
					euclid_list_kansei.append(euclid(human_vector, output_vector))
			euclid_list_all = np.array(euclid_list_all)
			euclid_list_normal = np.array(euclid_list_normal)
			euclid_list_kansei = np.array(euclid_list_kansei)
			euclid_list_kansei_tagi = np.array(euclid_list_kansei_tagi)

			print(f"{method}-all: {np.mean(euclid_list_all)}")
			print(euclid_list_all)
			put_list_into_file(euclid_list_all, method, "all")
			print(f"{method}-noramal: {np.mean(euclid_list_normal)}")
			print(euclid_list_normal)
			put_list_into_file(euclid_list_normal, method, "normal")
			print(f"{method}-kansei: {np.mean(euclid_list_kansei)}")
			print(euclid_list_kansei)
			put_list_into_file(euclid_list_kansei, method, "kansei")
			print(f"{method}-kansei_tagi: {np.mean(euclid_list_kansei_tagi)}")
			print(euclid_list_kansei_tagi)
			put_list_into_file(euclid_list_kansei_tagi, method, "kansei_tagi")
