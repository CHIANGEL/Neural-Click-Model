python -u run.py --train --optim adam --eval_freq 100 --check_point 100 \
--dataset demo \
--learning_rate 0.001 --lr_decay 0.7 --weight_decay 1e-5 --dropout_rate 0.5 \
--num_steps 40000 --embed_size 32 --hidden_size 64 --batch_size 32 --patience 5 \
--model_dir ./outputs/models/ \
--result_dir ./outputs/results/ \
--summary_dir ./outputs/summary/ \
--log_dir ./outputs/log/