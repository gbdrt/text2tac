
python new_datasets_hg_transformer_cl.py --train_dataset_file data/v15_partial_lemmavalsplit_training.txt --val_dataset_file data/v15_partial_lemmavalsplit_validation.txt --save_folder experiment1 --max_vocab_size 3000 --minimum_frequency 2 --embedding_size 768 --layers 12 --number_of_epochs 20000 --device_batch_size 8

