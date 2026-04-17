from hmmlearn import hmm
import numpy as np
import nltk

nltk.download('punkt')

text = "Ravi works at Infosys"

tokens = nltk.word_tokenize(text)

vocab = list(set(tokens))
word_to_id = {w:i for i,w in enumerate(vocab)}

X = np.array([word_to_id[w] for w in tokens]).reshape(-1,1)

model = hmm.MultinomialHMM(n_components=3, n_iter=20)
model.fit(X)

states = model.predict(X)

for w, s in zip(tokens, states):
    print(w, "->", s)
