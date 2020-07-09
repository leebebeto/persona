EXP_NAME=rho
TARGET_DATASET=preprocessed_lyrics
SOURCE_DATASET=filter_imdb
DA_NETWORK=DASTC
TARGET_DATASET_PORTION=1.0

RHO=1
CUDA_VISIBLE_DEVICES=0 python train_domain_adapt_js.py --domain_adapt \
--dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET} --network ${DA_NETWORK}\
 --training_portion ${TARGET_DATASET_PORTION} --save_model --save_samples \
 --rho $RHO --logDir 'logs/'$EXP_NAME$RHO

RHO=5
CUDA_VISIBLE_DEVICES=0 python train_domain_adapt_js.py --domain_adapt \
--dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET} --network ${DA_NETWORK}\
 --training_portion ${TARGET_DATASET_PORTION} --save_model --save_samples \
 --rho $RHO --logDir 'logs/'$EXP_NAME$RHO

RHO=10
CUDA_VISIBLE_DEVICES=0 python train_domain_adapt_js.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET} --network ${DA_NETWORK}\
 --training_portion ${TARGET_DATASET_PORTION} --save_model --save_samples \
 --rho $RHO --logDir 'logs/'$EXP_NAME$RHO

RHO=25
CUDA_VISIBLE_DEVICES=0 python train_domain_adapt_js.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET} --network ${DA_NETWORK}\
 --training_portion ${TARGET_DATASET_PORTION} --save_model --save_samples \
 --rho $RHO --logDir 'logs/'$EXP_NAME$RHO