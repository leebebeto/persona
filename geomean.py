import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('geomean', exist_ok = True)

pretrain_epoch = 5

def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum()/len(a))

loss_file = 'logs/imdb_lyrics_' + str(pretrain_epoch) +'.pickle'
text_file = 'transfer_list/sample_result_' + str(pretrain_epoch) +'.pickle'


with open(loss_file, 'rb') as f:
	data = pickle.load(f)
	data = data[:-1]

for item in data:
	bleu, style_acc = item['blue'], item['transfer_acc']
	geomean = geo_mean_overflow([bleu, style_acc])
	item['geomean'] = geomean

iter_list, max_list = [], []
max_geomean, cnt, patience = 0, 0, 30
early_stop_data = None
for i, item in enumerate(data):

	if item['epoch'] <= pretrain_epoch:
		continue

	if item['geomean'] > max_geomean:
		max_geomean = item['geomean']
		cnt = 0
	else:
		cnt += 1
		if cnt > patience:
			print('early stopping....')
			early_stop_data = item
			iter_list.append(i)
			max_list.append(max_geomean)
			break
	iter_list.append(float(i))
	max_list.append(float(max_geomean))


with open(text_file, 'rb') as f1:
	text_data = pickle.load(f1)

early_result = text_data[early_stop_data['epoch']]
original_result = text_data[20]

excel_data = [early_result, original_result]
df = pd.DataFrame(excel_data)
df = df.T

df.columns = [early_stop_data['epoch'], str(20)]

df.to_csv('geomean/geo_compare_'+str(pretrain_epoch)+'.csv')

# visualize max geomean plot
# plt.plot(iter_list, max_list)
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.title('Domain Accuracy')
# plt.savefig('geomean.png')
# plt.close()

