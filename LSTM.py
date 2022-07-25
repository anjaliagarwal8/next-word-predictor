from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
from doc3 import training_doc3
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras_preprocessing.sequence import pad_sequences
import nltk
nltk.download('punkt')

def preprocess(doc):
    cleaned = re.sub(r'\W+', ' ', doc).lower()
    tokens = word_tokenize(cleaned)
    train_len = 3+1
    text_sequences = []
    for i in range(train_len,len(tokens)):
        seq = tokens[i-train_len:i]
        text_sequences.append(seq)
    sequences = {}
    count = 1
    for i in range(len(tokens)):
        if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = tokenizer.texts_to_sequences(text_sequences) 
    
    #Collecting some information   
    vocabulary_size = len(tokenizer.word_counts)+1
    
    n_sequences = np.empty([len(sequences),train_len], dtype='int32')
    for i in range(len(sequences)):
        n_sequences[i] = sequences[i]
        
    train_inputs = n_sequences[:,:-1]
    train_targets = n_sequences[:,-1]
    train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
    seq_len = train_inputs.shape[1]
    
    return vocabulary_size, seq_len, train_inputs, train_targets, tokenizer

def model(vocab_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocab_size, seq_len, input_length=seq_len))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    
    return model
    
    
vocab_size, seq_len, train_inputs, train_targets, tokenizer = preprocess(training_doc3)
model = model(vocab_size, seq_len)
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_inputs,train_targets,epochs=500,verbose=1)
#model.save("mymodel.h5")
   
# Testing the model
input_text = input().strip().lower()
encoded_text = tokenizer.texts_to_sequences([input_text])[0]
pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
print(encoded_text, pad_encoded)
for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
  pred_word = tokenizer.index_word[i]
  print("Next word suggestion:",pred_word)