import PyPDF2
import re

pdf_path = "initiative.pdf"

text = ""
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()

sentences = re.split(r'[.!?]', text)

keywords = ["initiative", "scheme", "program", "mission", "project"]

print("Initiative Sentences:\n")

for s in sentences:
    if any(k in s.lower() for k in keywords):
        print("-", s.strip())
