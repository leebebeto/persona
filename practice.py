import pickle
with open('temp.pickle', 'rb') as f:
	temp = pickle.load(f)

print(temp)
#
#data = []
#
#for i in range(5):
#	temp = {'data': i}
#	data.append(temp)	
#	with open('temp.pickle', 'wb') as f:
#		pickle.dump(data, f)
#
