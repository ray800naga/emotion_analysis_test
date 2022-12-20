with open("/workspace/dataset/data_src/wiki40b_with_emotion/train/wakati/wiki_40b_train_with_emotion.txt", 'r') as f:
    count = 0
    for line in f:
        count = count + 1
    print("{}".format(count))