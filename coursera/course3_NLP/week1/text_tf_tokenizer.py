from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

senctences = [
    "I love my dog",
    "She loves her dog",
    "Do you love my dog?",
    "Obviously I love my dog!"
]

tokenizer = Tokenizer(
    num_words=20,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', # This is a default value and that's why in the out "?" or "!" doesn't appear
    lower=True, # This is also a default value and that's why the word_index doesn't have any words with capital letters
)

# Fit on text
tokenizer.fit_on_texts(senctences)
print(tokenizer.word_index)
# Output :
# {'dog': 1, 'love': 2, 'my': 3, 'i': 4, 'she': 5, 'loves': 6, 'her': 7, 'do': 8, 'you': 9, 'obviously': 10}

# Create sequence for already fit data
sequences = tokenizer.texts_to_sequences(
    senctences
)
print(sequences)
# Output : 
# [4, 2, 3, 1], [5, 6, 7, 1], [8, 9, 2, 3, 1], [10, 4, 2, 3, 1]]
# Comment : You will see how length of the list varies from sentence to sentence

# Create sequence for new data
new_unseen_sentences = [
    "Does anyone love their dog?",
    "Are you crazy",
    "I love my dog a lot!"
]
sequences = tokenizer.texts_to_sequences(
    new_unseen_sentences
)
print(sequences)
# Output :
# [[2, 1], [9], [4, 2, 3, 1]]
# Comment : New words are completly omitted with no trace to they being there

# New tokenizer with oov parameter
oov_tokenizer = Tokenizer(
    num_words=20,
    oov_token="<OOV>"
)
oov_tokenizer.fit_on_texts(senctences)
print(oov_tokenizer.word_index)
# Output : 
# {'<OOV>': 1, 'dog': 2, 'love': 3, 'my': 4, 'i': 5, 'she': 6, 'loves': 7, 'her': 8, 'do': 9, 'you': 10, 'obviously': 11}
sequences = oov_tokenizer.texts_to_sequences(
    new_unseen_sentences
)
print(sequences)
# Output : 
# [[1, 1, 3, 1, 2], [1, 10, 1], [5, 3, 4, 2, 1, 1]]
# Comment : Now even though the unseen data has new words it replaces unseen tokens with OOV token 

# Solve the uneven length of sequences by using padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequence = pad_sequences(
    sequences,
    maxlen=None, # Default value :: This will make the max length to be the longest sentence of the corpus
    padding='pre' # Default value :: padding will done before the sentence began 
)
print(padded_sequence)
# Output : 
'''
[[ 0  1  1  3  1  2]
 [ 0  0  0  1 10  1]
 [ 5  3  4  2  1  1]]
 '''
# Comment : This is for => "Does anyone love their dog?", "Are you crazy", "I love my dog a lot!"