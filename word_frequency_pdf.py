import PyPDF2
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')

def get_text(file):
    text = ""
    with open(file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

file1 = "file1.pdf"
file2 = "file2.pdf"

text1 = get_text(file1)
text2 = get_text(file2)

words1 = word_tokenize(text1.lower())
words2 = word_tokenize(text2.lower())

freq1 = Counter(words1)
freq2 = Counter(words2)

common = (freq1 + freq2).most_common(10)

print("Common Words:", common)

for w, _ in common:
    print(w, "Doc1:", freq1[w], "Doc2:", freq2[w])
