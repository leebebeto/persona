import pickle 
import csv
import pandas as pd

pretrain_epoch = 1
with open('transfer_list/sample_result_'+ str(pretrain_epoch) + '.pickle', 'rb') as f:
	data = pickle.load(f)
df = pd.DataFrame(data)
df = df.T

df.to_csv('transfer_list/sample_result_'+str(pretrain_epoch)+ '.csv')

# with open('sample_result_1.csv', 'wb') as file:
# 	writer = csv.writer(file)
# 	for key, value in data.items():
# 		import pdb; pdb.set_trace()
# 		writer.writerow([key, value])
