import pickle
import json 
import ast

with open('most_loved_lyric_list.pickle', 'rb') as file:
	data = pickle.load(file)

data = list(set(data))

new_data = [{"score": 1, "review": line} for line in data]

train_len = int(len(new_data) * 0.7)
valid_len = int(len(new_data) * 0.2)
# test_len = len(new_data) - (train_len + valid_len)

train_data = new_data[:train_len]
valid_data = new_data[train_len: train_len + valid_len]
test_data = new_data[train_len + valid_len: ]

with open('lyrics_train.txt', 'w') as file:
	for i in range(len(train_data)):
		temp = train_data[i]
		temp = json.dumps(ast.literal_eval(str(temp)))
		file.write(str(temp) + '\n')

with open('lyrics_valid.txt', 'w') as file:
	for i in range(len(valid_data)):
		temp = valid_data[i]
		temp = json.dumps(ast.literal_eval(str(temp)))
		file.write(str(temp) + '\n')

with open('lyrics_test.txt', 'w') as file:
	for i in range(len(test_data)):
		temp = test_data[i]
		temp = json.dumps(ast.literal_eval(str(temp)))
		file.write(str(temp) + '\n')
