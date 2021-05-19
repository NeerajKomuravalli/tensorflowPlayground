import json
from typing import Sequence
from numpy.lib.shape_base import tile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


headlines = []
sarcasm_label = []
article_url = []
with open("./Data/sarcasm.json") as file:
    data = json.load(file)

    for data_dict in data:
        headlines.append(data_dict["headline"])
        sarcasm_label.append(data_dict["is_sarcastic"])
        article_url.append(data_dict["article_link"])

for label, headline in zip(sarcasm_label[:5], headlines[:5]):
    print("{} :: {}".format(label, headline))

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(headlines)
# print(tokenizer.word_index)
sequences = tokenizer.texts_to_sequences(headlines)
padded_sequence = pad_sequences(sequences, padding="post")
print(padded_sequence[:5])