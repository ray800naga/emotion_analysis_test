f = open('./data/condition_nagasawa_test_EOS_remove_test/wakati_ja.wiki-1', 'r', encoding='utf-8')
# f = open('./data_src/ja.wiki-1')

count = 0
for line in f:
    print('count = ', count)
    print(line)
    count += 1
    if count == 300:
        break

f.close()