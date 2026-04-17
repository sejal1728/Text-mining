import nltk
import string
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

text = """Text Mining is an important process.
It helps in extracting useful information from text data.
Text mining includes tokenization, stopword removal, and more."""

print("Text:\n", text)

# Sentence Tokenization
sentences = nltk.sent_tokenize(text)
print("\nSentence Tokenization:", sentences)

# Word Tokenization
words = nltk.word_tokenize(text)
print("\nWord Tokenization:", words)

# Frequency Distribution
frequency = Counter(words)
print("\nFrequency Distribution:", frequency)

# Stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print("\nStopwords Sample:", stop_words[:10])

# Remove stopwords
filtered_words = [w for w in words if w.lower() not in stop_words]
print("\nAfter Stopword Removal:", filtered_words)

# Remove punctuation
clean_words = [w for w in filtered_words if w not in string.punctuation]
print("\nAfter Punctuation Removal:", clean_words)

# Final text
clean_text = " ".join(clean_words)
print("\nFinal Clean Text:", clean_text)
