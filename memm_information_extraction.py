from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

data = [
    [("John", "O"), ("Software", "JOB")],
    [("Alice", "O"), ("Data", "JOB")]
]

def features(sent, i):
    word = sent[i][0]
    return {"word": word.lower(), "cap": word[0].isupper()}

X, y = [], []

for sent in data:
    for i in range(len(sent)):
        X.append(features(sent, i))
        y.append(sent[i][1])

vec = DictVectorizer()
X = vec.fit_transform(X)

clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

print("Model trained")
