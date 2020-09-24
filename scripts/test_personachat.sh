cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

task_name=persona_chat
DATA_DIR=${parentdir}/data/personachat_processed

latest_run=`ls -dt runs/* |head -n 1`
latest_checkpoint=${latest_run}/checkpoints
# latest_checkpoint=runs/1600864341/checkpoints
echo $latest_checkpoint

test_file=$DATA_DIR/processed_test_self_original.txt    # for self_original
# test_file=$DATA_DIR/processed_test_self_revised.txt     # for self_revised
vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
output_file=./test_out_FIRE_personachat_original.txt

max_utter_num=15
max_utter_len=20
max_response_num=20
max_response_len=20
max_persona_num=5
max_persona_len=15
max_word_length=18
batch_size=32

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -u ${PKG_DIR}/model/eval.py \
                  --task_name $task_name \
                  --test_file $test_file \
                  --vocab_file $vocab_file \
                  --char_vocab_file $char_vocab_file \
                  --output_file $output_file \
                  --max_utter_num $max_utter_num \
                  --max_utter_len $max_utter_len \
                  --max_response_num $max_response_num \
                  --max_response_len $max_response_len \
                  --max_persona_num $max_persona_num \
                  --max_persona_len $max_persona_len \
                  --max_word_length $max_word_length \
                  --batch_size $batch_size \
                  --checkpoint_dir $latest_checkpoint > log_FIRE_test_personachat_original.txt 2>&1 &
