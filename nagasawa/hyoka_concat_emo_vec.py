# 実験1の実験データの結合

file_marks = ['B', 'C', 'D', 'A']

with open('/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_A.txt', 'r') as f_A:
		with open('/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_B.txt', 'r') as f_B:
			with open('/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_C.txt', 'r') as f_C:
				with open('/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_D.txt', 'r') as f_D:
					with open('/home/student/docker_share/dataset/data/hyoka/jikken_1/human/emotion_vector_all.txt', 'w') as f_w:
						for i in range(22):
							f_w.write(f_B.readline())
							f_w.write(f_C.readline())
							f_w.write(f_D.readline())
							f_w.write(f_A.readline())
