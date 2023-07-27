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
max_length = max([len(text) for text in text])
# One-hot encode the labels
one_hot_labels = to_categorical(padded_tokens)

# Split the data and labels into train and test sets

x_train, x_test, y_train, y_test = train_test_split(padded_tokens, one_hot_labels, test_size=0.2)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"x_train:{x_train}")
print(f"y_train:{y_train}")
print(f"x_test:{x_test}")
print(f"y_test:{y_test}")