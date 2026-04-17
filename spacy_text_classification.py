import spacy
from spacy.training.example import Example

nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")

textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

train_data = [
    ("I love this product", {"cats": {"POSITIVE": 1, "NEGATIVE": 0}}),
    ("This movie is great", {"cats": {"POSITIVE": 1, "NEGATIVE": 0}}),
    ("I hate this service", {"cats": {"POSITIVE": 0, "NEGATIVE": 1}}),
]

optimizer = nlp.begin_training()

for i in range(10):
    losses = {}
    for text, ann in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, ann)
        nlp.update([example], losses=losses)
    print("Epoch", i, losses)

test = nlp("The product is very good")
print("Prediction:", test.cats)
