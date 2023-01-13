import numpy as np

def get_top_k_indexes(list, K):
	sorted_list = sorted(list, reverse=True)
	top_k_indexes_list = []
	for i in range(K):
		top_k_indexes_list.append(list.index(sorted_list[i]))
	return top_k_indexes_list

method_list = ["BERT+weight", "BERT", "embedding"]
for method in method_list:
	with open("/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_all.txt", 'r') as f_human:
		with open(f"/home/student/docker_share/dataset/data/hyoka/jikken_1/{method}/output_emotion_vector_{method}.txt", 'r') as f_output:
			top_1_acc_list = []
			top_3_acc_list = []
			for i in range(88):
				human_vector = list(map(float, f_human.readline()[:-1].split('\t')[2:]))
				output_vector = list(map(float, f_output.readline()[1:-2].split(', ')))
				# human_vector = np.array(human_vector)
				# output_vector = np.array(output_vector)
				# print("human: ", human_vector)
				# print("output", output_vector)
				answer = human_vector.index(max(human_vector))
				# print("answer", answer)
				top_1_list = get_top_k_indexes(output_vector, 1)
				# print("top1: ", top_1_list)
				if answer in top_1_list:
					top_1_acc_list.append(True)
				else:
					top_1_acc_list.append(False)
				top_3_list = get_top_k_indexes(output_vector, 3)
				# print("top3: ", top_3_list)
				if answer in top_3_list:
					top_3_acc_list.append(True)
				else:
					top_3_acc_list.append(False)
			top_1_acc = top_1_acc_list.count(True) / len(top_1_acc_list) * 100
			top_3_acc = top_3_acc_list.count(True) / len(top_3_acc_list) * 100
			print(method)
			print("all")
			print(f"top_1_accuracy: {top_1_acc}%")
			print(f"top_3_accuracy: {top_3_acc}%")

			print("normal_words")
			top_1_acc = top_1_acc_list[:36].count(True) / len(top_1_acc_list[:36]) * 100
			top_3_acc = top_3_acc_list[:36].count(True) / len(top_3_acc_list[:36]) * 100
			print(f"top_1_accuracy: {top_1_acc}%")
			print(f"top_3_accuracy: {top_3_acc}%")

			print("kansei_tagi_words")
			top_1_acc = top_1_acc_list[36:70].count(True) / len(top_1_acc_list[36:70]) * 100
			top_3_acc = top_3_acc_list[36:70].count(True) / len(top_3_acc_list[36:70]) * 100
			print(f"top_1_accuracy: {top_1_acc}%")
			print(f"top_3_accuracy: {top_3_acc}%")

			print("kansei_words")
			top_1_acc = top_1_acc_list[70:].count(True) / len(top_1_acc_list[70:]) * 100
			top_3_acc = top_3_acc_list[70:].count(True) / len(top_3_acc_list[70:]) * 100
			print(f"top_1_accuracy: {top_1_acc}%")
			print(f"top_3_accuracy: {top_3_acc}%")
