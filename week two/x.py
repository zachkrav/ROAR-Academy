import os
import PyPDF2
import re
from collections import defaultdict

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_text(text):
    # Remove numbers, punctuation, and convert to lowercase
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()
    text = text.replace('httpswwwfulltextarchivecomsense', '') # Take out website from the text

def create_frequency_table(text):
    frequency_table = defaultdict(int)
    words = text.split()
    for word in words:
        if word.isalpha() and word.lower() not in ['chapter', 'page']:
            frequency_table[word] += 1
    return dict(frequency_table)

# Get PDF path
path = os.path.dirname(os.path.abspath(__file__))
pdf = path + '/' + 'Sense-and-Sensibility-by-Jane-Austen.pdf'


# Extract text from PDF
full_text = extract_text_from_pdf(pdf)

# Clean the text
cleaned_text = clean_text(full_text)

# Create frequency table
frequency_table = create_frequency_table(cleaned_text)



# Find the most frequent word
most_frequent_word = max(frequency_table, key=frequency_table.get)
most_frequent_count = frequency_table[most_frequent_word]

print(f"The most frequently used word is '{most_frequent_word}' with {most_frequent_count} occurrences.")