import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.preprocessing.sequence import pad_sequences

# Load the IMDB dataset
# Firstly the IMDB dataset contains only two columns, one is review text (content) and other is target 1 or 0 for determining positive or negative content review respectively. There is a voculabory dataset which contains the words with numbered integers. From that vocubulory the top most 10000 words are considered. The Xtrain or Xtest table contains the integer array corresponding to the words. If the words in the review are out of first 10000 words then they are replaced with special tokens.
num_words = 10000  # Vocabulary size, only consider the top 10,000 most common words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Preprocess the data
# As we know that the Xtrain or Xtest contains integer array in each row ,so here we are minimizing to select first 100 words only and if in the case array contains less than 100 then left over space filled with 0 in the beginning of array.
max_len = 100  # Maximum length of a review (truncate/pad reviews to this length)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Build the model
# 1.Primarily we initiated a sequential model and added embedded layer that converts the each word from the input to 64 dimensional array which help in finding the similar kind of words (for ex: super , excellent both have similar values in the 64D array).
# 2. Flatten method takes the input from previous layer and convert it into 1D array.
# 3. We added a dense layer with 128 units or neurons.This layer contains the weights and bias which are introduced during the training which help in finding the accurate solution to the model.Those values are updated frequently as the input sent into the model.
# 4. We had a output layer with one neuron which the decides the review positive or negative based on the value in that neuron. 
model = Sequential()
model.add(Embedding(num_words, 64, input_length=max_len))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Note : The values of weights and biases are not stored directly in the neurons themselves. Instead, they are stored as parameters within each layer of the neural network model

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Make predictions
def predict_sentiment(review):
    # Tokenize and pad the input review
    review_sequence = [imdb.get_word_index().get(word.lower(), 0) for word in review.split()]
    review_sequence = [word_index + 3 for word_index in review_sequence]
    review_sequence = pad_sequences([review_sequence], maxlen=max_len)
    # Predict sentiment (0 = negative, 1 = positive)
    sentiment = model.predict(review_sequence)[0][0]
    if sentiment >= 0.5:
        return "Positive"
    else:
        return "Negative"

# Example usage
review = "The movie was fantastic! Great acting and storyline."
print(f"Sentiment: {predict_sentiment(review)}")
