import pickle
import json 
import ast

with open('most_loved_lyric_list.pickle', 'rb') as file:
	love_data = pickle.load(file)

love_data = list(set(love_data))
love_data = [{"score": 1, "review": line} for line in love_data]


with open('hiphop.pickle', 'rb') as file:
	hiphop_data = pickle.load(file)

hiphop_data = list(set(hiphop_data))
hiphop_data = [{"score": 0, "review": line} for line in hiphop_data]

total_len = len(love_data) 
train_len = int(total_len * 0.7)
valid_len = int(total_len * 0.2)
# test_len = len(new_data) - (train_len + valid_len)

train_love = love_data[:train_len]
valid_love = love_data[train_len: train_len + valid_len]
test_love = love_data[train_len + valid_len: ]

total_len = len(hiphop_data) 
train_len = int(total_len * 0.7)
valid_len = int(total_len * 0.2)

train_hiphop = hiphop_data[:train_len]
valid_hiphop = hiphop_data[train_len: train_len + valid_len]
test_hiphop = hiphop_data[train_len + valid_len: ]

train_data = train_love + train_hiphop
valid_data = valid_love + valid_hiphop
test_data = test_love + test_hiphop

with open('data/lyrics/train/train.txt', 'w') as file:
	for i in range(len(train_data)):
		temp = train_data[i]
		temp = json.dumps(ast.literal_eval(str(temp)))
		file.write(str(temp) + '\n')

with open('data/lyrics/valid/valid.txt', 'w') as file:
	for i in range(len(valid_data)):
		temp = valid_data[i]
		temp = json.dumps(ast.literal_eval(str(temp)))
		file.write(str(temp) + '\n')

with open('data/lyrics/test/test.txt', 'w') as file:
	for i in range(len(test_data)):
		temp = test_data[i]
		temp = json.dumps(ast.literal_eval(str(temp)))
		file.write(str(temp) + '\n')
