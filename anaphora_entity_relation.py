import PyPDF2
import spacy

nlp = spacy.load("en_core_web_sm")

pdf_path = "sample.pdf"

text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for p in reader.pages:
        text += p.extract_text()

doc = nlp(text)

print("Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

print("\nRelations:")
subject = None
for token in doc:
    if token.dep_ == "nsubj":
        subject = token.text
    if token.dep_ == "dobj" and subject:
        print(subject, "->", token.text)
