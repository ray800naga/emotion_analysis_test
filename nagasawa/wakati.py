# %% [markdown]
# テキストの分かち書きを行う

# %%
import MeCab
import os
import tqdm
import re

# %%
# 入力されたテキストを分かち書きし、ノイズを除去してリスト化
def make_wakati_list_no_noize(str):
    # chasen: 品詞など、詳細情報を表示
    m_chasen = MeCab.Tagger()

    # wakati: シンプルに分かち書きしたものを出力
    m_wakati = MeCab.Tagger("-Owakati")

    # 分かち書き
    line_wakati = m_wakati.parse(str)
    # print('分かち書きした原文\n', line_wakati)

    # 記号・アルファベットの除去
    line_wakati = re.sub(re.compile("[!-/:-~]"), "", line_wakati)
    # print('記号除去\n', line_wakati)

    # 数字の除去
    line_wakati = re.sub(re.compile("[0-9]"), "", line_wakati)
    # print('数字除去\n', line_wakati)

    # 句読点の除去
    line_wakati = re.sub(re.compile("[。、]"), "", line_wakati)
    # print('句読点除去\n', line_wakati)

    # CR/LFの除去
    line_wakati = line_wakati.replace('\n', '').replace('\r', '')

    # リスト化
    wakati_list = line_wakati.split()
    # print('リスト\n', wakati_list)
    
    return wakati_list

# %%
condition = 'condition_nagasawa_test_EOS_remove_test'   # データ生成の条件
data_src = './data_src/'                # データ入力元
datadir = './data/' + condition + '/'   # データ出力先
print(os.getcwd())                      # 現在の作業ディレクトリ表示

# %%
target = 'ja.wiki-1'    # 分かち書きの対象となるファイル

# %%
# 残す品詞の指定
non_stop_word = ['動詞', '名詞', '形容詞', '形容動詞']

# ストップ品詞の指定
stop_jouken = ['代名詞', '接尾', '非自立', '数']

# ストップワードの指定
stop_word = ['する', 'いる', 'ある', 'なる', 'ない', 'できる', 'おり', '行う', 'ば', 'ら']

# %%
# chasen: 品詞など、詳細情報を表示
m_chasen = MeCab.Tagger()

# wakati: シンプルに分かち書きしたものを出力
m_wakati = MeCab.Tagger("-Owakati")

# コード内の文字エンコードで初期化される(?)　エラー回避のため
m_chasen.parse('')
m_wakati.parse('')

# %%
# 読み込むファイルを開く
f_src = open(data_src + 'ja.wiki-1', 'r', encoding='utf-8')

# 書き込むファイルを開く
f_dst = open(datadir + 'wakati_' + target, 'w', encoding='utf-8')

# %%
# ファイルの行数を取得
line_num = 60905911
"""for line in f_src:
    line_num += 1
print(line_num)"""

# %%
count = 0
for line in f_src:
    print("[{}%]".format(count / line_num * 100))
    count += 1
    wakati_list = make_wakati_list_no_noize(line)
    # あまりにも短い場合には処理をスキップ
    if len(wakati_list) <= 3:
        continue
    for word in wakati_list:
        # chasenの出力をリストに変換
        word_chasen = re.split('[,\t]', m_chasen.parse(word))
        # print(word_chasen)
        if len(word_chasen) >= 5:
            # 原型がstop_wordに含まれていない
            if word_chasen[7] not in stop_word:
                # print(word_chasen[1], word_chasen[2])
                if (word_chasen[1] in non_stop_word) and (word_chasen[2] not in stop_jouken):
                    f_dst.write(word_chasen[7] + ' ')
    f_dst.write('\n')
    # 原文も一緒に保存
    f_dst.write(line)
    f_dst.write('\n')

    # テスト用(50行で終了)
    # if count >= 50: # for debug
    #     break
# %%
f_src.close()
f_dst.close()

# %%



