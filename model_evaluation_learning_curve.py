import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

text = "Machine learning is fun. AI is powerful. Data science is useful."

sentences = text.split(".")

labels = [0 if i % 2 == 0 else 1 for i in range(len(sentences))]

vec = TfidfVectorizer()
X = vec.fit_transform(sentences)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
