import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os

'''----------------------------------------------------------------------------------------------
------------------------------------------BUILDING-DATASET---------------------------------------
----------------------------------------------------------------------------------------------'''

def build_dataset(words, n_words, atleast=1):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary



'''----------------------------------------------------------------------------------------------
-------------------------------------------CLEANING-DATA---------------------------------------------------------------------------------------------------------------------------------------'''
lines = open('data.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('data.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 10:
        id2line[_line[0]] = _line[4]
        
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
    
questions = []
print(questions)
answers = []
print(answers)

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return ' '.join([i.strip() for i in filter(None, text.split())])

clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))
    
min_line_length = 2
max_line_length = 5
short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

question_test = short_questions[500:550]
answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]


'''-----------------------------------------------------------------------------------------------------------------------------------------CONCATANATION---------------------------------------------------------------------------------------------------------------------------------------'''

#QUESTIONS 
concat_from = ' '.join(short_questions+question_test).split()
vocabulary_size_from = len(list(set(concat_from)))
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from)
print('vocab from size: %d'%(vocabulary_size_from))
print('Most common words', count_from[4:10])
print('Sample data', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print('filtered vocab size:',len(dictionary_from))
print("% of vocab used: {}%".format(round(len(dictionary_from)/vocabulary_size_from,4)*100))

#ANSWERS
concat_to = ' '.join(short_answers+answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print('vocab from size: %d'%(vocabulary_size_to))
print('Most common words', count_to[4:10])
print('Sample data', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print('filtered vocab size:',len(dictionary_to))
print("% of vocab used: {}%".format(round(len(dictionary_to)/vocabulary_size_to,4)*100))


GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']

for i in range(len(short_answers)):
    short_answers[i] += ' EOS'
    
'''----------------------------------------------------------------------------------------------
-------------------------------------------MODEL-----------------------------------------------------------------------------------------------------------------------------------------------'''

from tensorflow.python.util import nest
from tensorflow.python.layers.core import Dense

def gnmt_residual_fn(inputs, outputs):
    def split_input(inp, out):
        out_dim = out.get_shape().as_list()[-1]
        inp_dim = inp.get_shape().as_list()[-1]
        return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
    actual_inputs, _ = nest.map_structure(split_input, inputs, outputs)

    def assert_shape_match(inp, out):
        inp.get_shape().assert_is_compatible_with(out.get_shape())
    nest.assert_same_structure(actual_inputs, outputs)
    nest.map_structure(assert_shape_match, actual_inputs, outputs)
    return nest.map_structure(lambda inp, out: inp + out, actual_inputs, outputs)

class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):

    def __init__(self, attention_cell, cells, use_new_attention=True):
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(GNMTAttentionMultiCell, self).__init__(
            cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with bottom layer's attention copied to all upper layers."""
        if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(
                    inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):
                    cell = self._cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat(
                            [cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat(
                            [cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)

class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate, beam_width = 15):
        
        def cells(size,reuse=False):
            return tf.nn.rnn_cell.GRUCell(size,reuse=reuse)
        
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        decoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, decoder_input)
        
        num_residual_layer = num_layers - 2
        num_bi_layer = 1
        num_ui_layer = num_layers - num_bi_layer

        for n in range(num_bi_layer):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cells(size_layer),
                cell_bw = cells(size_layer),
                inputs = encoder_embedded,
                sequence_length = self.X_seq_len,
                dtype = tf.float32,
                scope = 'bidirectional_rnn_%d'%(n))
            encoder_embedded = tf.concat((out_fw, out_bw), 2)
        
        gru_cells = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_ui_layer)])
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                gru_cells,
                encoder_embedded,
                dtype=tf.float32,
                sequence_length=self.X_seq_len)
        
        encoder_state = (state_bw,) + (
                (encoder_state,) if num_ui_layer == 1 else encoder_state)
        
        decoder_cells = []
        for n in range(num_layers):
            cell = cells(size_layer)
            if (n >= num_layers - num_residual_layer):
                cell = tf.nn.rnn_cell.ResidualWrapper(cell, residual_fn = gnmt_residual_fn)
            decoder_cells.append(cell)
        attention_cell = decoder_cells.pop(0)
        to_dense = tf.layers.Dense(to_dict_size)
        
        with tf.variable_scope('decode'):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = size_layer, 
                memory = encoder_outputs,
                memory_sequence_length = self.X_seq_len)
            att_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = attention_cell,
                attention_mechanism = attention_mechanism,
                attention_layer_size = None,
                alignment_history = True,
                output_attention = False)
            gcell = GNMTAttentionMultiCell(att_cell, decoder_cells)
            
            self.initial_state = tuple(
                zs.clone(cell_state=es)
                if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
                for zs, es in zip(
                    gcell.zero_state(batch_size, dtype=tf.float32), encoder_state))
            
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_embedded,
                self.Y_seq_len,
                time_major = False
            )
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = gcell,
                helper = training_helper,
                initial_state = self.initial_state,
                output_layer = to_dense)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                maximum_iterations = tf.reduce_max(self.Y_seq_len))
            self.training_logits = training_decoder_output.rnn_output
            
        with tf.variable_scope('decode', reuse=True):
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(encoder_outputs, beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
            X_seq_len_tiled = tf.contrib.seq2seq.tile_batch(self.X_seq_len, beam_width)
            
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = size_layer, 
                memory = encoder_out_tiled,
                memory_sequence_length = X_seq_len_tiled)
            att_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = attention_cell,
                attention_mechanism = attention_mechanism,
                attention_layer_size = None,
                alignment_history = False,
                output_attention = False)
            gcell = GNMTAttentionMultiCell(att_cell, decoder_cells)
            
            self.initial_state = tuple(
                zs.clone(cell_state=es)
                if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
                for zs, es in zip(
                    gcell.zero_state(batch_size * beam_width, dtype=tf.float32), encoder_state_tiled))
            
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell = gcell,
                embedding = decoder_embeddings,
                start_tokens = tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
                end_token = EOS,
                initial_state = self.initial_state,
                beam_width = beam_width,
                output_layer = to_dense,
                length_penalty_weight = 0.0)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = False,
                maximum_iterations = 2 * tf.reduce_max(self.X_seq_len))
            self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]
            
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits,axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
'''----------------------------------------------------------------------------------------------
---------------------------------------APPLYING-PARAMETERS---------------------------------------
----------------------------------------------------------------------------------------------'''
size_layer = 128
num_layers = 4
embedded_size = 128
learning_rate = 0.001
batch_size = 16
epoch = 20

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, len(dictionary_from), 
                len(dictionary_to), learning_rate,batch_size)
sess.run(tf.global_variables_initializer())

def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
        X.append(ints)
    return X

'''-----------------------------------------------------------------------------------------------------------------------------------------TRIANING--------------------------------------------------------------------------------------------------------------------------------------------'''

#CREATING TRAIN AND TEST DATASETS
X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)

def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens
#EVALUTING
for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    X, Y = shuffle(X, Y)
    for k in range(0, len(short_questions), batch_size):
        index = min(k + batch_size, len(short_questions))
        batch_x, seq_x = pad_sentence_batch(X[k: index], PAD)
        batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD)
        predicted, accuracy, loss, _ = sess.run([model.predicting_ids,
                                      model.accuracy, model.cost, model.optimizer], 
                                      feed_dict={model.X:batch_x,
                                                model.Y:batch_y})
        total_loss += loss
        total_accuracy += accuracy
    total_loss /= (len(short_questions) / batch_size)
    total_accuracy /= (len(short_questions) / batch_size)
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))
    
batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD)
predicted = sess.run(model.predicting_ids, feed_dict={model.X:batch_x})

for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:',' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0,1,2,3]]))
    print('REAL ANSWER:',' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0,1,2,3]]))
    print('PREDICTED ANSWER:',' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0,1,2,3]]),'\n')