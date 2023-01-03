with open("/workspace/dataset/data_src/input_text.txt", 'r') as f_r:
	with open("/workspace/dataset/data_src/input_text_A.txt", 'w') as f_a:
		with open("/workspace/dataset/data_src/input_text_B.txt", 'w') as f_b:
			with open("/workspace/dataset/data_src/input_text_C.txt", 'w') as f_c:
				with open("/workspace/dataset/data_src/input_text_D.txt", 'w') as f_d:
					count = 0
					for line in f_r:
						count += 1
						
						if count % 4 == 0:
							f_a.write(line)
						elif count % 4 == 1:
							f_b.write(line)
						elif count % 4 == 2:
							f_c.write(line)
						elif count % 4 == 3:
							f_d.write(line)
