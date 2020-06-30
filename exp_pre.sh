device=$1
epoch=$2

CUDA_VISIBLE_DEVICES=$device python3 train_domain_adapt_js.py --domain_adapt --dataset preprocessed_lyrics --source_dataset filter_imdb --network DASTC --pretrain_epochs $epoch --save_model --save_samples --logDir 'logs/20'


