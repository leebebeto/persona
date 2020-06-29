epoch=$1

CUDA_VISIBLE_DEVICES=0 python3 train_domain_adapt_js.py --domain_adapt --dataset lyrics --source_dataset filter_imdb --network DASTC --pretrain_epochs $epoch



