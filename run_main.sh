





python main.py \
    --output_dir "/home/jsb193/workspace/[study]Attention/output_dir" \
    --cache_dir "/data/jsb193/[study]Attention/.cache" \
    --learning_rate "1e-5" \
    --per_device_train_batch_size "4" \
    --per_device_eval_batch_size "4" \
    --num_train_epochs "3" \
    --gradient_accumulation_steps "2" \
    --eval_accumulation_steps "2" \
    --lr_scheduler_type "linear" \
    --warmup_steps "1000" \
    --logging_strategy "steps" \
    --logging_steps "10" \
    --evaluation_strategy "steps" \
    --eval_steps "5000" \

