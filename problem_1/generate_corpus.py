import os
import re
import fitz  
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + " "
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url.strip(), headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        elements = soup.find_all(['p', 'li', 'h1', 'h2', 'h3'])
        return " ".join([elem.get_text(strip=True) for elem in elements])
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return clean_tokens

def generate_corpus_file(input_folder="content", output_filename="corpus.txt"):
    if not os.path.exists(input_folder):
        print(f"Error: Could not find the folder '{input_folder}'.")
        return

    print(f"Generating {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_folder):
            filepath = os.path.join(input_folder, filename)
            
            if filename.lower().endswith(".pdf"):
                print(f"Processing: {filename}")
                raw_text = extract_text_from_pdf(filepath)
                if raw_text:
                    tokens = preprocess_text(raw_text)
                    # Write the cleaned document as a single space-separated line
                    outfile.write(" ".join(tokens) + "\n")
                    
            elif filename.lower().endswith(".txt") and filename != output_filename:
                print(f"Processing URLs from: {filename}")
                with open(filepath, 'r', encoding='utf-8') as url_file:
                    urls = url_file.readlines()
                    
                for url in urls:
                    url = url.strip()
                    if url:
                        raw_text = extract_text_from_url(url)
                        if raw_text:
                            tokens = preprocess_text(raw_text)
                            outfile.write(" ".join(tokens) + "\n")

    print(f"Successfully created {output_filename}!")

if __name__ == "__main__":
    generate_corpus_file()