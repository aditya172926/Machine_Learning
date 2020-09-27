import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import *
from keras.initializers import *
import tensorflow as tf
import time, random

#Hyperparameters
batch_size = 64
latent_dimension = 256
num_samples = 10000

#Vectorise the data
input_texts = []
target_texts =[]
input_chars = set()
target_chars = set()

with open ('englist_to_french/fra.txt', 'r', encoding='utf-8') as f:
    print('File read')
    lines = f.read().split('\n')
    for line in lines[:min(num_samples, len(lines)-1)]:
        input_text, target_text = line.split('\t')
        target_text = '\t'+target_text+'\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_chars:
                input_chars.add(char)
        for char in target_text:
            if char not in target_chars:
                target_chars.add(char)

input_chars = sorted(list(input_chars))
target_chars = sorted(list(target_chars))
num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

#Print
# print('Number of samples:', len(input_texts))
# print('Number of unique input tokens:', num_encoder_tokens)
# print('Number of unique output tokens:', num_decoder_tokens)
# print('Max sequence length for inputs:', max_encoder_seq_length)
# print('Max sequence length for outputs:', max_decoder_seq_length)

#Define data for encoder and decoder

input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
target_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_in_data = np.zeros((len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_in_data[i, t, input_token_id[char]] = 1
    for t, char in enumerate(target_text):
        decoder_in_data[i, t, target_token_id[char]] = 1
        if t > 0:
            decoder_target_data[i, t - 1, target_token_id[char]] = 1

#Define and process input sequence
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dimension, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

#Using `encoder_states` set up the decoder as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dimension, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#Final model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())

#Model data Shape
# print("encoder_in_data shape:",encoder_in_data.shape)
# print("decoder_in_data shape:",decoder_in_data.shape)
# print("decoder_target_data shape:",decoder_target_data.shape)

#Compiling and training the model 
#FOR LARGE NUMBER OF EPOCHES A HIGH END MACHINE IS RECOMMENDED
# model.compile(optimizer='adam', loss='categorical_crossentropy')

# model.fit([encoder_in_data, decoder_in_data], decoder_target_data, batch_size = batch_size, epochs=50, validation_split=0.2)