import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import japanize_matplotlib

with open("/home/student/docker_share/dataset/data/model/512_400dim_MSE_window_3_relu.pth.log", 'r') as f:
	count = 0
	train_loss_list = []
	val_loss_list = []
	for line in f:
		count += 1
		if count <= 2:
			continue
		else:
			if count % 2 == 1:
				continue
			else:
				line_list = line.split()
				if(len(line_list) == 7):
					break
				train_loss_list.append(float(line_list[6][:-1]))
				val_loss_list.append(float(line_list[8]))
	print(train_loss_list)
	print(val_loss_list)
	epoch_num = len(train_loss_list)
	
	plt.plot(range(1, epoch_num + 1), train_loss_list, label="train loss")
	plt.plot(range(1, epoch_num + 1), val_loss_list, label="validation loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(loc="upper right")
	plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
	plt.savefig("./BERT.png")
	plt.clf()

with open("/home/student/docker_share/dataset/data/model/512_400dim_MSE_window_3_weight_relu.pth.log", 'r') as f:
	count = 0
	train_loss_list = []
	val_loss_list = []
	for line in f:
		count += 1
		if count <= 2:
			continue
		else:
			if count % 2 == 1:
				continue
			else:
				line_list = line.split()
				if(len(line_list) == 7):
					break
				train_loss_list.append(float(line_list[6][:-1]))
				val_loss_list.append(float(line_list[8]))
	print(train_loss_list)
	print(val_loss_list)
	epoch_num = len(train_loss_list)
	
	plt.plot(range(1, epoch_num + 1), train_loss_list, label="train loss")
	plt.plot(range(1, epoch_num + 1), val_loss_list, label="validation loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(loc="upper right")
	plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
	plt.savefig("./BERT+weight.png")
	plt.clf()

with open("/home/student/docker_share/dataset/data/model/False_512_400dim_MSE_window_3_relu.pth.log", 'r') as f:
	count = 0
	train_loss_list = []
	val_loss_list = []
	for line in f:
		count += 1
		if count <= 2:
			continue
		else:
			if count % 2 == 1:
				continue
			else:
				line_list = line.split()
				if(len(line_list) == 7):
					break
				train_loss_list.append(float(line_list[6][:-1]))
				val_loss_list.append(float(line_list[8]))
	print(train_loss_list)
	print(val_loss_list)
	epoch_num = len(train_loss_list)
	
	plt.plot(range(1, epoch_num + 1), train_loss_list, label="train loss")
	plt.plot(range(1, epoch_num + 1), val_loss_list, label="validation loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(loc="upper right")
	plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
	plt.savefig("./embeddings.png")
	plt.clf()