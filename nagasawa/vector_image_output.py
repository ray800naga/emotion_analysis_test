import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from my_module.tools import get_emotion_word_dict
import numpy as np

def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i,y_list[i-1],"{:.3f}".format(y_list[i-1]), ha="center")

# 出力グラフ設定
left = np.array([i for i in range(1, 11)])
height = ([0.2 * i for i in range(1, 6)])
emotion_label = ["喜", "怒", "哀", "怖", "恥", "好", "厭", "昴", "安", "驚"]
emotion_word_dict = get_emotion_word_dict()
word_list = ["気持ち", "涙", "思い"]
for word in word_list:
	vector = emotion_word_dict[word]
	plt.bar(left, vector, tick_label=emotion_label)
	plt.ylim(0, 1)
	add_value_label(left, vector)
	plt.title(word)
	plt.savefig("/workspace/vector_{}.png".format(word))
	plt.clf()
