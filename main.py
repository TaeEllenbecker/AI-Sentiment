import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer  # For tokenizing the text
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences
import pandas as pd  # For data manipulation (although not used directly here)
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Load the IMDB dataset
# data set contains 1 for postive and 0 for negative
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)
# num_words=10000 limits the dataset to the 10,000 most common words
# train_data contains 25,000 lists, each list contains an array of integers corresponding to a word in the data set (each integer is one of 10,000 words)
# each list is a movie review 

# word_index is now a dictionary of integers -> words for ease of reading
word_index = tf.keras.datasets.imdb.get_word_index()

# Create a reversed dictionary to decode the integers back to words
reverse_word_index = {value: key for (key, value) in word_index.items()}


# Decode the first training review back to text... train_data[0] is the first review
# This will join the decoded words together
# the ? in this will be returned if i cannot be found in the dictionary
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in train_data[0]])
# Print the review
print(decoded_review)

# Set the maximum length for padded sequences
max_length = 250  # Define the maximum length of sequences (reviews)

# Pad and truncate the sequences to ensure all reviews have the same length for model input
train_data = pad_sequences(train_data, maxlen=max_length)  # Pad training data
test_data = pad_sequences(test_data, maxlen=max_length)    # Pad testing data


# above is known
# summary of above:
# set train_data and train_labels equal to training data and corresponding values
# set test_data and test_labels to the training data and training values
# created dictionary of integers and words
# reversed to make easier to read
# decoded the first review in train_data
#   the ? is the default if i is not present in the data set
# printed the review


#done



# Create the neural network model using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_length),  # Embedding layer for word representation... 16 dimensions

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # Bidirectional LSTM layer to capture dependencies in data

    tf.keras.layers.Dense(64, activation='relu'),  # Dense layer with ReLU activation
    
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model with loss function, optimizer, and metrics
model.compile(loss='binary_crossentropy',  # Loss function for binary classification
              optimizer='adam',  # Adam optimizer
              metrics=['accuracy'])  # Track accuracy during training

# Train the model on the training data
history = model.fit(train_data, train_labels,  # Training data and labels
                    epochs=10,  # Number of training epochs
                    batch_size=512,  # Size of training batches
                    validation_data=(test_data, test_labels),  # Validate on the test data
                    verbose=1)  # Print progress during training

# Evaluate the model on the test data
results = model.evaluate(test_data, test_labels)
print(f'Test Loss: {results[0]} - Test Accuracy: {results[1]}')  # Print test loss and accuracy

# # Store training history for plotting
# history_dict = history.history

# # Plot training and validation accuracy values
# plt.plot(history_dict['accuracy'])  # Plot training accuracy
# plt.plot(history_dict['val_accuracy'])  # Plot validation accuracy
# plt.title('Model accuracy')  # Set title for the plot
# plt.ylabel('Accuracy')  # Label for y-axis
# plt.xlabel('Epoch')  # Label for x-axis
# plt.legend(['Train', 'Test'], loc='upper left')  # Legend to distinguish between train and test accuracy
# plt.show()  # Show the plot

# # Plot training and validation loss values
# plt.plot(history_dict['loss'])  # Plot training loss
# plt.plot(history_dict['val_loss'])  # Plot validation loss
# plt.title('Model loss')  # Set title for the plot
# plt.ylabel('Loss')  # Label for y-axis
# plt.xlabel('Epoch')  # Label for x-axis
# plt.legend(['Train', 'Test'], loc='upper left')  # Legend to distinguish between train and test loss
# plt.show()  # Show the plot
# Save the trained model to a file
model.save("sentiment_model.h5")  # Saves the model to a file

