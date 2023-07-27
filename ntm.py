###--------------------------------------------data cleanin-----------------------------------###
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import re
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
        
def clean_text(text):
    text = " ".join(text).lower()
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

# Load your text data
input_data = open('data.txt', 'r', encoding = "utf8").read()
text = clean_text(input_data)
# Tokenize the text
tokens = word_tokenize(text)
vocab_size = len(Counter(tokens))

# Stem the tokens
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in stemmed_tokens if token not in stop_words]

# Tokenization
tokenizer  = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(filtered_tokens)
numerical_data = tokenizer.texts_to_sequences(filtered_tokens)

# Pad the filtered tokens
input_size = 14  # Define the fixed length of input sequence
padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(numerical_data, maxlen=input_size, padding='post', truncating='post')
print(len(padded_tokens))
max_length = max([len(text) for text in text])
# Split the data into training and test sets
train_data = padded_tokens[:int(len(padded_tokens) * 0.8)]
val_data = padded_tokens[int(len(padded_tokens) * 0.8):int(len(padded_tokens) * 0.9)]
test_data = padded_tokens[int(len(padded_tokens) * 0.9):]

'''
# Create the x_train, y_train, x_val, y_val, x_test, y_test variables
x_train = train_data[:,:-1]
y_train = train_data[:,-1]
x_val = val_data[:,:-1]
y_val = val_data[:,-1]
x_test = test_data[:,:-1]
y_test = test_data[:,-1]
labels = np.arange(len(padded_tokens))
'''
# One-hot encode the labels
one_hot_labels = to_categorical(padded_tokens)

# Split the data and labels into train and test sets
x_train, x_test, y_train, y_test = train_test_split(padded_tokens, one_hot_labels, test_size=0.2, random_state=42)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train)
print(y_train)

#x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)
# Define the external memory matrix
# Define the external memory matrix
memory_size = 64
memory_dim = 64
embedding_dim = 128
input_shape = 512
memory = tf.Variable(tf.random.normal([memory_size, memory_dim], stddev=0.1))

# Define the input and output for the network
inputs = tf.keras.layers.Input(shape=(input_shape,))
x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_shape)(inputs)
x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
x = tf.keras.layers.LSTM(128)(x)
keys = tf.keras.layers.Dense(memory_dim, activation='relu', kernel_initializer='he_normal')(x)
keys = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(keys)

# Define the read and write heads
read_weights = tf.keras.layers.Dense(memory_size, activation='softmax', kernel_initializer='glorot_normal')(keys)
write_weights = tf.keras.layers.Dense(memory_size, activation='sigmoid', kernel_initializer='glorot_normal')(keys)

# Perform a dot product between the read weights and the memory matrix
read_vectors = tf.keras.layers.Dot(axes=[-1,-1])([read_weights, memory])

# Perform an element-wise multiplication between the write weights and the memory matrix
erase_weights = tf.keras.layers.Dense(memory_dim, activation='sigmoid', kernel_initializer='glorot_normal')(keys)
add_weights = tf.keras.layers.Dense(memory_dim, activation='tanh', kernel_initializer='glorot_normal')(keys)

# Update the memory matrix
memory = memory * (1 - tf.keras.layers.Dot(axes=[1,2])([tf.expand_dims(write_weights, axis=2), tf.expand_dims(erase_weights, axis=1)])) + tf.matmul(tf.expand_dims(write_weights, axis=1), tf.expand_dims(add_weights, axis=2))

# Concatenate the read vectors and the keys
output = tf.keras.layers.Concatenate()([tf.squeeze(read_vectors), keys])
output = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(output)
output = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(output)
output = tf.keras.layers.Dense(64, activation='softmax')(output)

# Create the NTM model
model = tf.keras.Model(inputs=inputs, outputs=output)
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define the batch size and number of epochs
epochs = 10

# Use the fit method to train the NTM on the big data with batch processing
for e in range(epochs):
    print("Epoch: ", e)

    model.fit(x_train, y_train, epochs=10, batch_size=32)
             # validation_data=(x_test, y_test))

    # Evaluate the model on test data after each epoch
   # results = model.evaluate(x_test, y_test, batch_size=batch_size,x_val, y_val )
   # print('Test loss:', results[0])
    #print('Test accuracy:', results[1])

tf.keras.utils.plot_model(model, 'NTM_model.png', show_shapes=True)
