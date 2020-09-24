import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def get_char_embedding(charVocab):
    print("get_char_embedding")
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0
    return tf.constant(embeddings, name="word_char_embedding")

def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 
    return embeddings

def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

def cnn_layer(inputs, filter_sizes, num_filters, scope=None, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[2].value

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(i)):
                w = tf.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
            outputs.append(pooled)
    return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]

def my_softmax_loss(logits, labels, depth):
    # establish label smoothing
    smooth = tf.ones([1, depth], tf.float32) * 0.05
    labels_one_hot = tf.one_hot(labels, depth=depth, dtype=tf.float32) + smooth
    logits_scaled = tf.nn.softmax(logits, -1)
    cross_entropy_loss = - tf.reduce_sum(labels_one_hot*tf.log(logits_scaled), 1)
    return cross_entropy_loss


class FIRE(object):
    def __init__(
      self, max_utter_num, max_utter_len, max_response_num, max_response_len, max_persona_num, max_persona_len, 
        vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab, gamma, num_loop, l2_reg_lambda=0.0):

        self.utterances = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances")
        self.utterances_len = tf.placeholder(tf.int32, [None, max_utter_num], name="utterances_len")
        self.utterances_num = tf.placeholder(tf.int32, [None], name="utterances_num")
        self.u_charVec = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len, maxWordLength], name="utterances_char")
        self.u_charLen = tf.placeholder(tf.int32, [None, max_utter_num, max_utter_len], name="utterances_char_len")

        self.responses = tf.placeholder(tf.int32, [None, max_response_num, max_response_len], name="responses")
        self.responses_len = tf.placeholder(tf.int32, [None, max_response_num], name="responses_len")
        self.r_charVec = tf.placeholder(tf.int32, [None, max_response_num, max_response_len, maxWordLength], name="responses_char")
        self.r_charLen =  tf.placeholder(tf.int32, [None, max_response_num, max_response_len], name="responses_char_len")

        self.personas = tf.placeholder(tf.int32, [None, max_persona_num, max_persona_len], name="personas")
        self.personas_len = tf.placeholder(tf.int32, [None, max_persona_num], name="personas_len")
        self.personas_num = tf.placeholder(tf.int32, [None], name="personas_num")
        self.p_charVec = tf.placeholder(tf.int32, [None, max_persona_num, max_persona_len, maxWordLength], name="personas_char")
        self.p_charLen =  tf.placeholder(tf.int32, [None, max_persona_num, max_persona_len], name="personas_char_len")
        
        self.target = tf.placeholder(tf.int64, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(1.0)

        # =============================== Embedding Layer ===============================
        # word embedding
        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            utterances_embedded = tf.nn.embedding_lookup(W, self.utterances)  # [batch_size, max_utter_num, max_utter_len, word_dim]
            responses_embedded = tf.nn.embedding_lookup(W, self.responses)    # [batch_size, max_response_num, max_response_len, word_dim]
            personas_embedded = tf.nn.embedding_lookup(W, self.personas)      # [batch_size, max_persona_num, max_persona_len, word_dim]
            print("original utterances_embedded: {}".format(utterances_embedded.get_shape()))
            print("original responses_embedded: {}".format(responses_embedded.get_shape()))
            print("original personas_embedded: {}".format(personas_embedded.get_shape()))
        
        with tf.name_scope('char_embedding'):
            char_W = get_char_embedding(charVocab)
            utterances_char_embedded = tf.nn.embedding_lookup(char_W, self.u_charVec)  # [batch_size, max_utter_num, max_utter_len,  maxWordLength, char_dim]
            responses_char_embedded = tf.nn.embedding_lookup(char_W, self.r_charVec)   # [batch_size, max_response_num, max_response_len, maxWordLength, char_dim]
            personas_char_embedded = tf.nn.embedding_lookup(char_W, self.p_charVec)    # [batch_size, max_persona_num, max_persona_len, maxWordLength, char_dim]
            print("utterances_char_embedded: {}".format(utterances_char_embedded.get_shape()))
            print("responses_char_embedded: {}".format(responses_char_embedded.get_shape()))
            print("personas_char_embedded: {}".format(personas_char_embedded.get_shape()))

        char_dim = utterances_char_embedded.get_shape()[-1].value
        utterances_char_embedded = tf.reshape(utterances_char_embedded, [-1, maxWordLength, char_dim])  # [batch_size*max_utter_num*max_utter_len, maxWordLength, char_dim]
        responses_char_embedded = tf.reshape(responses_char_embedded, [-1, maxWordLength, char_dim])    # [batch_size*max_response_num*max_response_len, maxWordLength, char_dim]
        personas_char_embedded = tf.reshape(personas_char_embedded, [-1, maxWordLength, char_dim])      # [batch_size*max_persona_num*max_persona_len, maxWordLength, char_dim]

        # char embedding
        utterances_cnn_char_emb = cnn_layer(utterances_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=False) # [batch_size*max_utter_num*max_utter_len, emb]
        cnn_char_dim = utterances_cnn_char_emb.get_shape()[1].value
        utterances_cnn_char_emb = tf.reshape(utterances_cnn_char_emb, [-1, max_utter_num, max_utter_len, cnn_char_dim])                                # [batch_size, max_utter_num, max_utter_len, emb]

        responses_cnn_char_emb = cnn_layer(responses_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=True)    # [batch_size*max_response_num*max_response_len,  emb]
        responses_cnn_char_emb = tf.reshape(responses_cnn_char_emb, [-1, max_response_num, max_response_len, cnn_char_dim])                            # [batch_size, max_response_num, max_response_len, emb]

        personas_cnn_char_emb = cnn_layer(personas_char_embedded, filter_sizes=[3, 4, 5], num_filters=50, scope="CNN_char_emb", scope_reuse=True)      # [batch_size*max_persona_num*max_persona_len,  emb]
        personas_cnn_char_emb = tf.reshape(personas_cnn_char_emb, [-1, max_persona_num, max_persona_len, cnn_char_dim])                                # [batch_size, max_persona_num, max_persona_len, emb]
                
        utterances_embedded = tf.concat(axis=-1, values=[utterances_embedded, utterances_cnn_char_emb])   # [batch_size, max_utter_num, max_utter_len, emb]
        responses_embedded  = tf.concat(axis=-1, values=[responses_embedded, responses_cnn_char_emb])     # [batch_size, max_response_num, max_response_len, emb]
        personas_embedded  = tf.concat(axis=-1, values=[personas_embedded, personas_cnn_char_emb])        # [batch_size, max_persona_num, max_persona_len, emb]
        utterances_embedded = tf.nn.dropout(utterances_embedded, keep_prob=self.dropout_keep_prob)
        responses_embedded = tf.nn.dropout(responses_embedded, keep_prob=self.dropout_keep_prob)
        personas_embedded = tf.nn.dropout(personas_embedded, keep_prob=self.dropout_keep_prob)
        print("utterances_embedded: {}".format(utterances_embedded.get_shape()))
        print("responses_embedded: {}".format(responses_embedded.get_shape()))
        print("personas_embedded: {}".format(personas_embedded.get_shape()))


        # =============================== Encoding Layer ===============================
        with tf.variable_scope("encoding_layer") as vs:
            
            emb_dim = utterances_embedded.get_shape()[-1].value
            parallel_utterances_embedded = tf.reshape(utterances_embedded, [-1, max_utter_len, emb_dim])  # [batch_size*max_utter_num, max_utter_len, emb]
            parallel_utterances_len = tf.reshape(self.utterances_len, [-1])                               # [batch_size*max_utter_num, ]
            parallel_responses_embedded = tf.reshape(responses_embedded, [-1, max_response_len, emb_dim]) # [batch_size*max_response_num, max_response_len, emb]
            parallel_responses_len = tf.reshape(self.responses_len, [-1])                                 # [batch_size*max_response_num, ]
            parallel_personas_embedded = tf.reshape(personas_embedded, [-1, max_persona_len, emb_dim])    # [batch_size*max_persona_num, max_persona_len, emb]
            parallel_personas_len = tf.reshape(self.personas_len, [-1])                                   # [batch_size*max_persona_num, ]

            rnn_scope_name = "bidirectional_rnn"
            u_rnn_output, u_rnn_states = lstm_layer(parallel_utterances_embedded, parallel_utterances_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
            utterances_output = tf.concat(axis=2, values=u_rnn_output)  # [batch_size*max_utter_num, max_utter_len, rnn_size*2]
            r_rnn_output, r_rnn_states = lstm_layer(parallel_responses_embedded, parallel_responses_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            responses_output = tf.concat(axis=2, values=r_rnn_output)   # [batch_size*max_response_num, max_response_len, rnn_size*2]
            p_rnn_output, p_rnn_states = lstm_layer(parallel_personas_embedded, parallel_personas_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)
            personas_output = tf.concat(axis=2, values=p_rnn_output)    # [batch_size*max_persona_num, max_persona_len, rnn_size*2]
            print("encoded utterances : {}".format(utterances_output.shape))
            print("encoded responses : {}".format(responses_output.shape))
            print("encoded personas : {}".format(personas_output.shape))

            output_dim = utterances_output.get_shape()[-1].value
            flatten_utterances_output = tf.reshape(utterances_output, [-1, max_utter_num*max_utter_len, output_dim])
            flatten_personas_output = tf.reshape(personas_output, [-1, max_persona_num*max_persona_len, output_dim])

            mask_u = tf.sequence_mask(parallel_utterances_len, max_utter_len, dtype=tf.float32)     # [batch_size*max_utter_num, max_utter_len]
            mask_u = tf.reshape(mask_u, [-1, max_utter_num*max_utter_len])                          # [batch_size, max_utter_num*max_utter_len]
            mask_u_1 = tf.expand_dims(mask_u, -1)                                                   # [batch_size, max_utter_num*max_utter_len, 1]
            mask_u_2 = tf.expand_dims(tf.expand_dims(mask_u, 1), 1)                                 # [batch_size, 1, 1, max_utter_num*max_utter_len]

            mask_r = tf.sequence_mask(parallel_responses_len, max_response_len, dtype=tf.float32)   # [batch_size*max_response_num, max_response_len]
            mask_r = tf.reshape(mask_r, [-1, max_response_num, max_response_len])                   # [batch_size, max_response_num, max_response_len]
            mask_r = tf.expand_dims(mask_r, -1)                                                     # [batch_size, max_response_num, max_response_len, 1]

            mask_p = tf.sequence_mask(parallel_personas_len, max_persona_len, dtype=tf.float32)     # [batch_size*max_persona_num, max_persona_len]
            mask_p = tf.reshape(mask_p, [-1, max_persona_num*max_persona_len])                      # [batch_size, max_persona_num*max_persona_len]
            mask_p_1 = tf.expand_dims(mask_p, 1)                                                    # [batch_size, 1, max_persona_num*max_persona_len]
            mask_p_2 = tf.expand_dims(tf.expand_dims(mask_p, 1), 1)                                 # [batch_size, 1, 1, max_persona_num*max_persona_len]
        

        # =============================== Filtering Layer ===============================
        with tf.variable_scope("filtering_layer") as vs:

            # context and knowledge filters
            similarity_UP = tf.matmul(flatten_utterances_output,  # [batch_size, max_utter_num*max_utter_len, max_persona_num*max_persona_len]
                                      flatten_personas_output, transpose_b=True, name='similarity_matrix_UP')

            similarity_UP_mask_u = similarity_UP * mask_u_1 + -1e9 * (1-mask_u_1)                                # [batch_size, max_utter_num*max_utter_len, max_persona_num*max_persona_len]
            attention_weight_for_u_up = tf.nn.softmax(tf.transpose(similarity_UP_mask_u, perm=[0,2,1]), dim=-1)  # [batch_size, max_persona_num*max_persona_len, max_utter_num*max_utter_len]
            attended_personas_output_up = tf.matmul(attention_weight_for_u_up, flatten_utterances_output)        # [batch_size, max_persona_num*max_persona_len, dim]

            similarity_UP_mask_p = similarity_UP * mask_p_1 + -1e9 * (1-mask_p_1)                                # [batch_size, max_utter_num*max_utter_len, max_persona_num*max_persona_len]
            attention_weight_for_p_up = tf.nn.softmax(similarity_UP_mask_p, dim=-1)                              # [batch_size, max_utter_num*max_utter_len, max_persona_num*max_persona_len]
            attended_utterances_output_up = tf.matmul(attention_weight_for_p_up, flatten_personas_output)        # [batch_size, max_utter_num*max_utter_len, dim]

            m_u_up = tf.concat(axis=-1, values=[flatten_utterances_output, attended_utterances_output_up, tf.multiply(flatten_utterances_output, attended_utterances_output_up), flatten_utterances_output-attended_utterances_output_up]) # [batch_size, max_utter_num*max_utter_len, dim]
            m_p_up = tf.concat(axis=-1, values=[flatten_personas_output, attended_personas_output_up, tf.multiply(flatten_personas_output, attended_personas_output_up), flatten_personas_output-attended_personas_output_up])             # [batch_size, max_persona_num*max_persona_len, dim]
            concat_dim = m_u_up.get_shape()[-1].value

            match_w_up = tf.get_variable("match_w_up", [concat_dim, output_dim])
            match_b_up = tf.get_variable("match_b_up", [output_dim])
            flatten_utterances_output = tf.nn.relu(tf.einsum('aij,jk->aik', m_u_up, match_w_up) + match_b_up) + flatten_utterances_output
            flatten_personas_output = tf.nn.relu(tf.einsum('aij,jk->aik', m_p_up, match_w_up) + match_b_up) + flatten_personas_output

            utterances_vec = tf.reshape(tf.concat(axis=1, values=[u_rnn_states[0].h, u_rnn_states[1].h]), [-1, max_utter_num, output_dim])  # [batch_size, max_utter_num, rnn_size*2]
            personas_vec = tf.reshape(tf.concat(axis=1, values=[p_rnn_states[0].h, p_rnn_states[1].h]), [-1, max_persona_num, output_dim])  # [batch_size, max_persona_num, rnn_size*2]

            M_sent = tf.get_variable("M_sent", shape=[output_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
            s_sent = tf.einsum('aij,jk->aik', personas_vec, M_sent)                      # [batch_size, max_persona_num, rnn_size*2]
            s_sent = tf.matmul(s_sent, utterances_vec, transpose_b=True, name="s_sent")  # [batch_size, max_persona_num, max_utter_num]

            # Max aggregate
            s_sent = s_sent * tf.expand_dims(tf.sequence_mask(self.utterances_num, max_utter_num, dtype=tf.float32), 1) # [batch_size, max_persona_num, max_utter_num]
            match_score_p = tf.reshape(tf.reduce_max(s_sent, -1), [-1, 1, 1])               # [batch_size*max_persona_num, 1, 1]

            refer_mask_p = tf.cast(tf.greater(tf.sigmoid(match_score_p), gamma), "float")   # [batch_size*max_persona_num, 1, 1]
            mask_p_2 = mask_p_2 * tf.reshape(tf.tile(refer_mask_p, [1, max_persona_len, 1]), [-1, 1, 1, max_persona_num*max_persona_len])  #[batch_size, 1, 1, max_persona_num*max_persona_len]

            print("establish the context and knowledge filters")


        # =============================== Iteratively Referring Layer ===============================
        tiled_utterances_output = tf.tile(tf.expand_dims(flatten_utterances_output, 1), [1, max_response_num, 1, 1])  # [batch_size, max_response_num, max_utter_num*max_utter_len, rnn_size*2]
        responses_output = tf.reshape(responses_output, [-1, max_response_num, max_response_len, output_dim])         # [batch_size, max_response_num, max_response_len, rnn_size*2]
        tiled_personas_output = tf.tile(tf.expand_dims(flatten_personas_output, 1), [1, max_response_num, 1, 1])      # [batch_size, max_response_num, max_persona_num*max_persona_len, rnn_size*2]

        logits_list = []
        y_pred_list = []
        losses_list = []
        cur_utterances_output_ur = tiled_utterances_output  # [batch_size, max_response_num, max_utter_num*max_utter_len, rnn_size*2]
        cur_responses_output_ur = responses_output          # [batch_size, max_response_num, max_response_len, rnn_size*2]
        cur_personas_output_pr = tiled_personas_output      # [batch_size, max_response_num, max_persona_num*max_persona_len, rnn_size*2]
        cur_responses_output_pr = responses_output          # [batch_size, max_response_num, max_response_len, rnn_size*2]
        
        for loop in range(num_loop):

            with tf.variable_scope("loop_{}".format(loop)):
                # =============================== Matching Layer ===============================
                with tf.variable_scope("matching_layer") as vs:
                    # matching between context and response
                    similarity_UR = tf.matmul(cur_responses_output_ur,  # [batch_size, max_response_num, response_len, max_utter_num*max_utter_len]
                                              tf.transpose(cur_utterances_output_ur, perm=[0,1,3,2]), name='similarity_matrix_UR')

                    similarity_UR_mask_r = similarity_UR * mask_r + -1e9 * (1-mask_r)                                      # [batch_size, max_response_num, max_response_len, max_utter_num*max_utter_len]
                    attention_weight_for_r_ur = tf.nn.softmax(tf.transpose(similarity_UR_mask_r, perm=[0,1,3,2]), dim=-1)  # [batch_size, max_response_num, max_utter_num*max_utter_len, max_response_len]
                    attended_utterances_output_ur = tf.matmul(attention_weight_for_r_ur, cur_responses_output_ur)          # [batch_size, max_response_num, max_utter_num*max_utter_len, dim]

                    similarity_UR_mask_u = similarity_UR * mask_u_2 + -1e9 * (1-mask_u_2)                         # [batch_size, max_response_num, response_len, max_utter_num*max_utter_len]
                    attention_weight_for_u_ur = tf.nn.softmax(similarity_UR_mask_u, dim=-1)                       # [batch_size, max_response_num, response_len, max_utter_num*max_utter_len]
                    attended_responses_output_ur = tf.matmul(attention_weight_for_u_ur, cur_utterances_output_ur) # [batch_size, max_response_num, response_len, dim]

                    m_u_ur = tf.concat(axis=-1, values=[cur_utterances_output_ur, attended_utterances_output_ur, tf.multiply(cur_utterances_output_ur, attended_utterances_output_ur), cur_utterances_output_ur-attended_utterances_output_ur]) # [batch_size, max_response_num, max_utter_num*max_utter_len, dim]
                    m_r_ur = tf.concat(axis=-1, values=[cur_responses_output_ur, attended_responses_output_ur, tf.multiply(cur_responses_output_ur, attended_responses_output_ur), cur_responses_output_ur-attended_responses_output_ur])       # [batch_size, max_response_num, response_len, dim]
                    concat_dim = m_u_ur.get_shape()[-1].value
                    m_u_ur = tf.reshape(m_u_ur, [-1, max_utter_len, concat_dim])    # [batch_size*max_response_num*max_utter_num, max_utter_len, dim]
                    m_r_ur = tf.reshape(m_r_ur, [-1, max_response_len, concat_dim]) # [batch_size*max_response_num, max_response_len, dim]

                    match_w_ur = tf.get_variable("match_w_ur", [concat_dim, output_dim])
                    match_b_ur = tf.get_variable("match_b_ur", [output_dim])
                    next_utterances_output = tf.nn.relu(tf.einsum('aij,jk->aik', m_u_ur, match_w_ur) + match_b_ur)
                    next_responses_output = tf.nn.relu(tf.einsum('aij,jk->aik', m_r_ur, match_w_ur) + match_b_ur)
                    
                    if loop == 0:
                        cur_utterances_output_ur = tf.reshape(next_utterances_output, [-1, max_response_num, max_utter_num*max_utter_len, output_dim]) + cur_utterances_output_ur
                        cur_responses_output_ur = tf.reshape(next_responses_output, [-1, max_response_num, max_response_len, output_dim]) + cur_responses_output_ur
                    else:
                        cur_utterances_output_ur = tf.reshape(next_utterances_output, [-1, max_response_num, max_utter_num*max_utter_len, output_dim]) + cur_utterances_output_ur + tiled_utterances_output
                        cur_responses_output_ur = tf.reshape(next_responses_output, [-1, max_response_num, max_response_len, output_dim]) + cur_responses_output_ur + responses_output

                    
                    # matching between persona and response
                    similarity_PR = tf.matmul(cur_responses_output_pr,  # [batch_size, max_response_num, response_len, max_persona_num*max_persona_len]
                                              tf.transpose(cur_personas_output_pr, perm=[0,1,3,2]), name='similarity_matrix_PR')
                    
                    similarity_PR_mask_r = similarity_PR * mask_r + -1e9 * (1-mask_r)                                      # [batch_size, max_response_num, max_response_len, max_persona_num*max_persona_len]
                    attention_weight_for_r_pr = tf.nn.softmax(tf.transpose(similarity_PR_mask_r, perm=[0,1,3,2]), dim=-1)  # [batch_size, max_response_num, max_persona_num*max_persona_len, response_len]
                    attended_personas_output_pr = tf.matmul(attention_weight_for_r_pr, cur_responses_output_pr)            # [batch_size, max_response_num, max_persona_num*max_persona_len, dim]

                    similarity_PR_mask_p = similarity_PR * mask_p_2 + -1e9 * (1-mask_p_2)                        # [batch_size, max_response_num, response_len, max_persona_num*max_persona_len]
                    attention_weight_for_p_pr = tf.nn.softmax(similarity_PR_mask_p, dim=-1)                      # [batch_size, max_response_num, response_len, max_persona_num*max_persona_len]
                    attended_responses_output_pr = tf.matmul(attention_weight_for_p_pr, cur_personas_output_pr)  # [batch_size, max_response_num, response_len, dim]

                    m_p_pr = tf.concat(axis=-1, values=[cur_personas_output_pr, attended_personas_output_pr, tf.multiply(cur_personas_output_pr, attended_personas_output_pr), cur_personas_output_pr-attended_personas_output_pr])        # [batch_size, max_response_num, max_persona_num*max_persona_len, dim]
                    m_r_pr = tf.concat(axis=-1, values=[cur_responses_output_pr, attended_responses_output_pr, tf.multiply(cur_responses_output_pr, attended_responses_output_pr), cur_responses_output_pr-attended_responses_output_pr])  # [batch_size, max_response_num, response_len, dim]
                    m_p_pr = tf.reshape(m_p_pr, [-1, max_persona_len, concat_dim])   # [batch_size*max_response_num*max_persona_num, max_persona_len, dim]
                    m_r_pr = tf.reshape(m_r_pr, [-1, max_response_len, concat_dim])  # [batch_size*max_response_num, max_response_len, dim]

                    match_w_pr = tf.get_variable("match_w_pr", [concat_dim, output_dim])
                    match_b_pr = tf.get_variable("match_b_pr", [output_dim])
                    next_personas_output = tf.nn.relu(tf.einsum('aij,jk->aik', m_p_pr, match_w_pr) + match_b_pr)
                    next_responses_output = tf.nn.relu(tf.einsum('aij,jk->aik', m_r_pr, match_w_pr) + match_b_pr)
                    
                    if loop == 0:
                        cur_personas_output_pr = tf.reshape(next_personas_output, [-1, max_response_num, max_persona_num*max_persona_len, output_dim]) + cur_personas_output_pr
                        cur_responses_output_pr = tf.reshape(next_responses_output, [-1, max_response_num, max_response_len, output_dim]) + cur_responses_output_pr
                    else:
                        cur_personas_output_pr = tf.reshape(next_personas_output, [-1, max_response_num, max_persona_num*max_persona_len, output_dim]) + cur_personas_output_pr + tiled_personas_output
                        cur_responses_output_pr = tf.reshape(next_responses_output, [-1, max_response_num, max_response_len, output_dim]) + cur_responses_output_pr + responses_output


                # =============================== Aggregation Layer ===============================
                with tf.variable_scope("aggregation_layer") as vs:

                    utterances_output_cross_ur = cur_utterances_output_ur
                    responses_output_cross_ur = cur_responses_output_ur
                    personas_output_cross_pr = cur_personas_output_pr
                    responses_output_cross_pr = cur_responses_output_pr

                    # establish RNN aggregation on context and response
                    # aggregate utterance across utterance_len
                    utterances_output_cross_ur = tf.reshape(utterances_output_cross_ur, [-1, max_utter_len, output_dim])
                    final_utterances_max = tf.reduce_max(utterances_output_cross_ur, axis=1)                   # [batch_size*max_response_num*max_utter_num, 2*rnn_size]
                    final_utterances_mean = tf.reduce_mean(utterances_output_cross_ur, axis=1)                 # [batch_size*max_response_num*max_utter_num, 2*rnn_size]
                    final_utterances = tf.concat(axis=1, values=[final_utterances_max, final_utterances_mean]) # [batch_size*max_response_num*max_utter_num, 4*rnn_size]

                    # aggregate utterance across utterance_num
                    final_utterances = tf.reshape(final_utterances, [-1, max_utter_num, output_dim*2])         # [batch_size*max_response_num, max_utter_num, 4*rnn_size]
                    tiled_utters_num = tf.reshape(tf.tile(tf.expand_dims(self.utterances_num, 1), [1, max_response_num]), [-1, ])  # [batch_size*max_response_num, ]
                    rnn_scope_aggre = "bidirectional_rnn_aggregation"
                    final_utterances_output, final_utterances_state = lstm_layer(final_utterances, tiled_utters_num, rnn_size, self.dropout_keep_prob, rnn_scope_aggre, scope_reuse=False)
                    final_utterances_output = tf.concat(axis=2, values=final_utterances_output)                                    # [batch_size*max_response_num, max_utter_num, 2*rnn_size]
                    final_utterances_max = tf.reduce_max(final_utterances_output, axis=1)                                          # [batch_size*max_response_num, 2*rnn_size]
                    final_utterances_state = tf.concat(axis=1, values=[final_utterances_state[0].h, final_utterances_state[1].h])  # [batch_size*max_response_num, 2*rnn_size]
                    aggregated_utterances_ur = tf.concat(axis=1, values=[final_utterances_max, final_utterances_state])            # [batch_size*max_response_num, 4*rnn_size]

                    # aggregate response across response_len
                    responses_output_cross_ur = tf.reshape(responses_output_cross_ur, [-1, max_response_len, output_dim])
                    final_responses_max_ur = tf.reduce_max(responses_output_cross_ur, axis=1)                             # [batch_size*max_response_num, 2*rnn_size]
                    final_responses_mean_ur = tf.reduce_mean(responses_output_cross_ur, axis=1)                           # [batch_size*max_response_num, 2*rnn_size]
                    aggregated_responses_ur = tf.concat(axis=1, values=[final_responses_max_ur, final_responses_mean_ur]) # [batch_size*max_response_num, 4*rnn_size]


                    # establish ATT aggregation on persona and response
                    # aggregate persona across persona_len
                    personas_output_cross_pr = tf.reshape(personas_output_cross_pr, [-1, max_persona_len, output_dim])
                    final_personas_max = tf.reduce_max(personas_output_cross_pr, axis=1)                  # [batch_size*max_response_num*max_persona_num, 2*rnn_size]
                    final_personas_mean = tf.reduce_mean(personas_output_cross_pr, axis=1)                # [batch_size*max_response_num*max_persona_num, 2*rnn_size]
                    final_personas = tf.concat(axis=1, values=[final_personas_max, final_personas_mean])  # [batch_size*max_response_num*max_persona_num, 4*rnn_size]
                    final_personas = tf.reshape(final_personas, [-1, max_persona_num, output_dim*2])      # [batch_size*max_response_num, max_persona_num, 4*rnn_size]

                    # aggregate persona across persona_num
                    pers_w = tf.get_variable("pers_w", [output_dim*2, 1], initializer=tf.contrib.layers.xavier_initializer())
                    pers_b = tf.get_variable("pers_b", shape=[1, ], initializer=tf.zeros_initializer())
                    pers_weights = tf.nn.relu(tf.einsum('aij,jk->aik', final_personas, pers_w) + pers_b)                           # [batch_size*max_response_num, max_persona_num, 1]
                    
                    mask_p_num = tf.reshape(tf.sequence_mask(self.personas_num, max_persona_num, dtype=tf.float32), [-1, 1, 1])    # [batch_size*max_persona_num, 1, 1]
                    mask_p_num = mask_p_num * refer_mask_p
                    tile_mask_p_num = tf.reshape(tf.tile(tf.reshape(mask_p_num, [-1, 1, max_persona_num, 1]), [1, max_response_num, 1, 1]), 
                                                 [-1, max_persona_num, 1])                                                         # [batch_size*max_response_num, max_persona_num, 1]

                    pers_weights = pers_weights * tile_mask_p_num + -1e9 * (1-tile_mask_p_num)                                     # [batch_size*max_response_num, max_persona_num, 1]
                    pers_weights = tf.nn.softmax(pers_weights, dim=1)
                    aggregated_personas_pr = tf.matmul(tf.transpose(pers_weights, [0, 2, 1]), final_personas)  # [batch_size*max_response_num, 1, 4*rnn_size]
                    aggregated_personas_pr = tf.squeeze(aggregated_personas_pr, [1])                           # [batch_size*max_response_num, 4*rnn_size]

                    # aggregate response across response_len
                    responses_output_cross_pr = tf.reshape(responses_output_cross_pr, [-1, max_response_len, output_dim])
                    final_responses_max_pr = tf.reduce_max(responses_output_cross_pr, axis=1)                               # [batch_size*max_response_num, 2*rnn_size]
                    final_responses_mean_pr = tf.reduce_mean(responses_output_cross_pr, axis=1)                             # [batch_size*max_response_num, 2*rnn_size]
                    aggregated_responses_pr = tf.concat(axis=1, values=[final_responses_max_pr, final_responses_mean_pr])   # [batch_size*max_response_num, 4*rnn_size]

                    joined_feature =  tf.concat(axis=1, values=[aggregated_utterances_ur, aggregated_responses_ur, aggregated_personas_pr, aggregated_responses_pr]) # [batch_size*max_response_num, 16*rnn_size(3200)]


                # =============================== Prediction Layer ===============================
                with tf.variable_scope("prediction_layer"):
                    hidden_output_size = 256
                    regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
                    joined_feature = tf.nn.dropout(joined_feature, keep_prob=self.dropout_keep_prob)
                    full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                                    activation_fn=tf.nn.relu,
                                                                    reuse=False,
                                                                    trainable=True,
                                                                    scope="projected_layer")      # [batch_size*max_response_num, hidden_output_size(256)]
                    full_out = tf.nn.dropout(full_out, keep_prob=self.dropout_keep_prob)

                    bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
                    s_w = tf.get_variable("s_w", shape=[hidden_output_size, 1], initializer=tf.contrib.layers.xavier_initializer())

                    logits = tf.reshape(tf.matmul(full_out, s_w) + bias, [-1, max_response_num])  # [batch_size, max_response_num]
                    probs = tf.nn.softmax(logits, name="prob")                                    # [batch_size, max_response_num]
                    losses = my_softmax_loss(logits=logits, labels=self.target, depth=max_response_num)
                    # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
                    losses = tf.reduce_mean(losses, name="losses") + l2_reg_lambda * l2_loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                    logits_list.append(logits)
                    y_pred_list.append(probs)
                    losses_list.append(losses)

        print("establish {}-loop iteratively referring between context-response and persona-response".format(num_loop))
            

        # =============================== Final Prediction Layer ===============================
        with tf.variable_scope("prediction_layer") as vs:
            
            LOCAL_LOSS_FLAG = True
            if LOCAL_LOSS_FLAG:
                self.mean_loss = sum(losses_list)
                self.probs = tf.div(tf.add_n(y_pred_list), len(y_pred_list), name="prob")
                print("establish local loss")
            else:
                logits_sum = tf.add_n(logits_list)
                losses = my_softmax_loss(logits=logits_sum, labels=self.target, depth=max_response_num)
                # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_sum, labels=self.target)
                self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.probs = tf.nn.softmax(logits_sum, name="prob")
                print("establish global loss")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
