import pickle
import matplotlib.pyplot as plt
import os

os.makedirs('vis', exist_ok = True)
os.makedirs('transfer_list', exist_ok = True)

pretrain_epoch = 1

# load losses
with open('logs/' + 'imdb_lyrics_' + str(pretrain_epoch) + '.pickle', 'rb') as f:
	data = pickle.load(f)

iter_list = [float(item['iteration']) for item in data][:-1]
domain_list = [float(item['domain_acc']) for item in data][:-1]
transfer_list = [float(item['transfer_acc']) for item in data][:-1]
bleu_list = [float(item['blue']) for item in data][:-1]

plt.plot(iter_list, domain_list)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Domain Accuracy')
plt.savefig('vis/pretrain_epoch_'+str(pretrain_epoch)+'_domain_acc' +'.png')
plt.close()

plt.plot(iter_list, transfer_list)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Transfer Accuracy')
plt.savefig('vis/pretrain_epoch_'+str(pretrain_epoch)+'_transfer_acc' +'.png')
plt.close()

plt.plot(iter_list, bleu_list)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Bleu Score')
plt.savefig('vis/pretrain_epoch_'+str(pretrain_epoch)+'_bleu_score' +'.png')
plt.close()


recon_dict = {}
transfer_dict = {}

sample_sentences=[
	'but i can not help it that i can not take my eyes off of you',
	'and I am a man of expanding so why should i stand in her way',
	'and i swear by the moon and the stars in the sky ill be there',
	'baby I am baby I am hurt and i do not want to play anymore',
	'then I am willing to wait for it wait for it wait for it',
	'every inch of your skin is a holy gray I have got to find',
	'i never thought that you would be the one to hold my heart',
	'if she changes her mind this is the first place she will go',
	'oh her eyes her eyes make the stars look like they are not shinin']

total_dict = {}
for i in range(1,21):
	original_result, reconstruction_result, transfer_result =[], [], []  
	select_epoch = i
	# load sample sentences
	with open('logs/'+ str(pretrain_epoch) + '/' + 'domain_adapt/target/epoch'+ str(select_epoch) + '_reconstruction.txt', 'r') as f1:
		reconstruction = f1.readlines()
	with open('logs/'+ str(pretrain_epoch) + '/' + 'domain_adapt/target/epoch'+ str(select_epoch) + '_transfer.txt', 'r') as f2:
		transfer = f2.readlines()
	for recon in reconstruction:
		if recon.split('\t')[0] in sample_sentences:
			original_result.append(recon.split('\t')[0]) 
			reconstruction_result.append(recon.split('\t')[1]) 

	for trans in transfer:
		if trans.split('\t')[0] in sample_sentences:
			transfer_result.append(trans.split('\t')[1]) 

	total_dict[i] = {'original': original_result, 'recon': reconstruction_result, 'transfer': transfer_result}
	

with open('transfer_list/sample_result_' + str(pretrain_epoch) + '.pickle', 'wb') as result:
	pickle.dump(total_dict, result)
	








