python3 LSP_train.py \
--model_name_or_path configs/sentiment_da_model_mean_blend \
--train_input_file data/daily_dialog_sent_da_train.128len.db \
--eval_input_file data/daily_dialog_sent_da_val.tsv \
--init_checkpoint "models/small/small_ft.pkl" \
--output_dir "models/sentiment_da_model_mean_blend" \
--seed 42 \
--max_seq_length 128 \
--train_batch_size 256 \
--gradient_accumulation_steps 8 \
--eval_batch_size 64 \
--learning_rate 1e-5 \
--num_optim_steps 10000 \
--valid_step 2500 \
--warmup_steps 4000 \
--normalize_data true \
--fp16 false \
--lr_schedule noam \
--skip_eval \
--loss_scale 0.0 \
--no_token_id true \
--pbar true \
--working_pals 0 1 2 3 4 5 6 7 8 9 10 11 \
--trainable_tasks 0 1 \
--default_branch_train_prob 0.2

