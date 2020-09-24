cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

task_name=persona_chat
DATA_DIR=${parentdir}/data/personachat_processed

# for self_original
train_file=$DATA_DIR/processed_train_self_original.txt
valid_file=$DATA_DIR/processed_valid_self_original.txt
# for self_revised
# train_file=$DATA_DIR/processed_train_self_revised.txt
# valid_file=$DATA_DIR/processed_valid_self_revised.txt

vocab_file=$DATA_DIR/vocab.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
embedded_vector_file=$DATA_DIR/glove_42B_300d_vec_plus_word2vec_100.txt

max_utter_num=15
max_utter_len=20
max_response_num=20
max_response_len=20
max_persona_num=5
max_persona_len=15
max_word_length=18
embedding_dim=400
rnn_size=200
gamma=0.3
num_loop=3

batch_size=16
starter_learning_rate=0.00025
lambda=0
dropout_keep_prob=0.8
num_epochs=15
evaluate_every=500

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -u ${PKG_DIR}/model/train.py \
                --task_name $task_name \
                --train_file $train_file \
                --valid_file $valid_file \
                --vocab_file $vocab_file \
                --char_vocab_file $char_vocab_file \
                --embedded_vector_file $embedded_vector_file \
                --max_utter_num $max_utter_num \
                --max_utter_len $max_utter_len \
                --max_response_num $max_response_num \
                --max_response_len $max_response_len \
                --max_persona_num $max_persona_num \
                --max_persona_len $max_persona_len \
                --max_word_length $max_word_length \
                --embedding_dim $embedding_dim \
                --rnn_size $rnn_size \
                --gamma $gamma \
                --num_loop $num_loop \
                --batch_size $batch_size \
                --starter_learning_rate $starter_learning_rate \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --num_epochs $num_epochs \
                --evaluate_every $evaluate_every > log_FIRE_train_personachat_original.txt 2>&1 &
