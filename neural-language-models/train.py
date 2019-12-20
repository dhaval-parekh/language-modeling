
# Imports
import os
import tensorflow as tf

keras = tf.keras

# source text
data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """

# integer encode text
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([data])

# Covert each unique work into integer.
# So we process with integer number not with word itself.
encoded = tokenizer.texts_to_sequences([data])[0]

# Determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))

# split into X and y elements
# sequences = [sequences]
x = sequences[:0]
y = sequences[:1]
print(sequences)
print(x)
print(y)
print(vocab_size)
# one hot encode outputs
y = keras.utils.to_categorical(y, num_classes=vocab_size)

# define model
model = keras.models.Sequential()

# Add layers for neural networks.
model.add(keras.layers.Embedding(vocab_size, 10, input_length=1))
model.add(keras.layers.LSTM(50))
model.add(keras.layers.Dense(vocab_size, activation='softmax'))

print(model.summary())
