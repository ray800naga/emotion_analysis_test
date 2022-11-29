mode_list = ["train", "val", "test"]
for mode in mode_list:
    with open("/workspace/dataset/data_src/BERT_to_emotion/window_size_3/BERT_to_emo_{}.txt".format(mode), 'r') as f:
        count = 0
        for line in f:
            count = count + 1
        print("{}:{}".format(mode, count))