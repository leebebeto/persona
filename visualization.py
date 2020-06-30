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

#
#recon_dict = {}
#transfer_dict = {}
#for i in range(1,21):
#	select_epoch = i
## load sample sentences
#	with open('logs/'+ str(pretrain_epoch) + '/' + 'domain_adapt/target/epoch'+ str(select_epoch) + '_reconstruction.txt', 'rb') as f1:
#		reconstruction = f1.readlines()
#	recon_dict[i] = reconstruction
#
#	with open('logs/'+ str(pretrain_epoch) + '/' + 'domain_adapt/target/epoch'+ str(select_epoch) + '_transfer.txt', 'rb') as f2:
#		transfer = f2.readlines()
#	transfer_dict[i] = transfer
#
#
#sample_index = 
#
#import pdb; pdb.set_trace()
#
#




