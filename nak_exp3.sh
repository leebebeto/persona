EXP_NAME=party
TARGET_DATASET=preprocessed_tweets
SOURCE_DATASET=filter_imdb
DA_NETWORK=DASTC
TARGET_DATASET_PORTION=1.0


CUDA_VISIBLE_DEVICES=0 python train_domain_classifier.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET}

RHO=1
CUDA_VISIBLE_DEVICES=0 python train_domain_adapt_js.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET} --network ${DA_NETWORK}\
 --training_portion ${TARGET_DATASET_PORTION} --save_model --save_samples --rho $RHO --logDir 'logs/'$EXP_NAME$RHO --custom_styler_path $EXP_NAME$RHO
