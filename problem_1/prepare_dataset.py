import os
import re
import fitz  
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

# Download the NLTK tokenizer model (only needs to run once)
nltk.download('punkt', quiet=True)
# Optional: Download stopwords to make your Word Cloud more meaningful
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- 1. EXTRACTION FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + " "
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_text_from_url(url):
    """Fetches and extracts text from a webpage."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url.strip(), headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs, list items, and headings
        elements = soup.find_all(['p', 'li', 'h1', 'h2', 'h3'])
        return " ".join([elem.get_text(strip=True) for elem in elements])
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

# --- 2. PREPROCESSING FUNCTION ---

def preprocess_text(text):
    """Cleans, lowercases, filters, and tokenizes the text."""
    # Step 1: Lowercase the text
    text = text.lower()
    
    # Step 2: Remove non-English characters, punctuation, and numbers.
    # The regex [^a-z\s] matches anything that is NOT a lowercase english letter or whitespace.
    # Replacing these with a space effectively removes punctuation, numbers, and other languages (like Hindi).
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra spaces created by the previous step
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 3: Tokenization
    tokens = word_tokenize(text)
    
    # Step 4: Remove common English stopwords so they don't dominate the word cloud
    # (Words like 'the', 'and', 'is', 'of')
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return clean_tokens

# --- 3. MAIN EXECUTION PIPELINE ---

def main():
    folder_path = "content"
    all_corpus_tokens = []
    total_documents = 0
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Could not find the folder '{folder_path}'.")
        return

    print("Starting data extraction and preparation...\n")

    # Iterate through files in the content folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Handle PDFs
        if filename.lower().endswith(".pdf"):
            print(f"Processing PDF: {filename}")
            raw_text = extract_text_from_pdf(filepath)
            if raw_text:
                tokens = preprocess_text(raw_text)
                all_corpus_tokens.extend(tokens)
                total_documents += 1
                
        # Handle the text file containing URLs
        elif filename.lower().endswith(".txt"):
            print(f"Found URL list: {filename}")
            with open(filepath, 'r', encoding='utf-8') as file:
                urls = file.readlines()
                
            for url in urls:
                url = url.strip()
                if url: # Skip empty lines
                    print(f"  Fetching URL: {url}")
                    raw_text = extract_text_from_url(url)
                    if raw_text:
                        tokens = preprocess_text(raw_text)
                        all_corpus_tokens.extend(tokens)
                        total_documents += 1

    # --- 4. STATISTICS AND REPORTING ---
    
    total_tokens = len(all_corpus_tokens)
    vocabulary_size = len(set(all_corpus_tokens))
    
    print("\n" + "="*30)
    print("DATASET STATISTICS")
    print("="*30)
    print(f"Total Number of Documents Processed: {total_documents}")
    print(f"Total Number of Tokens: {total_tokens}")
    print(f"Vocabulary Size (Unique Tokens): {vocabulary_size}")
    
    # --- 5. WORD CLOUD GENERATION ---
    
    if total_tokens > 0:
        print("\nGenerating Word Cloud...")
        # Join tokens back into a single string for the word cloud
        text_for_cloud = " ".join(all_corpus_tokens)
        
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white',
                              colormap='viridis',
                              max_words=100).generate(text_for_cloud)
        
        # Display the generated image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Most Frequent Words in IIT Jodhpur Corpus", fontsize=16)
        
        # Save the image to your folder
        plt.savefig("wordcloud_output.png")
        print("Word cloud saved as 'wordcloud_output.png'.")
        
        plt.show()

        output_file = "cleaned_tokens.json"
        with open(output_file, "w") as f:
            json.dump(all_corpus_tokens, f)
        print(f"\nSaved {len(all_corpus_tokens)} tokens to {output_file}")

    else:
        print("\nNo tokens were extracted. Check your files and URLs.")

if __name__ == "__main__":
    main()