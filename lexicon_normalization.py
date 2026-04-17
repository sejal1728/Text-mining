import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "Ratan Tata founded Tata Group in India and he is leading business growth"

tokens = word_tokenize(text)

print("Tokens:", tokens)

# Stemming
stemmer = PorterStemmer()
print("\nStemming:", [stemmer.stem(t) for t in tokens])

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("\nLemmatization:", [lemmatizer.lemmatize(t) for t in tokens])

# POS Tagging
pos_tags = pos_tag(tokens)
print("\nPOS Tags:", pos_tags)

# NER
ner_tree = ne_chunk(pos_tags)
print("\nNER Tree:\n", ner_tree)
