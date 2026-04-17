import spacy

nlp = spacy.load("en_core_web_sm")

text = "Ravi works at Infosys and Priya works at TCS"
doc = nlp(text)

for sent in doc.sents:
    person = None
    org = None
    for ent in sent.ents:
        if ent.label_ == "PERSON":
            person = ent.text
        if ent.label_ == "ORG":
            org = ent.text
    if person and org:
        print(person, "-> works at ->", org)
