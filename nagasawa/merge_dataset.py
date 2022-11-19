# 分割されたデータセットを1ファイルにまとめる
import os

root_dirname = "/wordspace/dataset/data_src/"
file_names = [os.path.join(root_dirname, n) for n in os.listdir(root_dirname)]
