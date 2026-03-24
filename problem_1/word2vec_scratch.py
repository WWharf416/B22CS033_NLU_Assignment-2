import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
import os
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- 1. DATASET & VOCABULARY PREPARATION ---

class Word2VecData:
    def __init__(self, tokens, min_count=5):
        self.tokens = tokens
        self.build_vocab(min_count)
        self.init_negative_sampling_distribution()
        
    def build_vocab(self, min_count):
        print("Building vocabulary...")
        word_counts = Counter(self.tokens)
        self.vocab = [word for word, count in word_counts.items() if count >= min_count]
        
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.vocab)
        
        self.data = [self.word2idx[word] for word in self.tokens if word in self.word2idx]
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Total words in processed data: {len(self.data)}")

    def init_negative_sampling_distribution(self):
        counts = np.array([self.tokens.count(self.idx2word[i]) for i in range(self.vocab_size)])
        freqs = counts ** 0.75
        self.sampling_prob = freqs / sum(freqs)

    def get_negative_samples(self, num_samples):
        return np.random.choice(self.vocab_size, size=num_samples, p=self.sampling_prob, replace=True).tolist()

    def generate_batches(self, window_size, num_neg_samples, batch_size, is_cbow=False):
        targets, contexts, negatives = [], [], []
        
        for i in range(window_size, len(self.data) - window_size):
            center_word = self.data[i]
            context_words = self.data[i - window_size : i] + self.data[i + 1 : i + window_size + 1]
            
            if is_cbow:
                targets.append(center_word)
                contexts.append(context_words)
                negatives.append(self.get_negative_samples(num_neg_samples))
            else:
                for context_word in context_words:
                    targets.append(center_word)
                    contexts.append(context_word)
                    negatives.append(self.get_negative_samples(num_neg_samples))
                    
            if len(targets) >= batch_size:
                yield torch.tensor(targets), torch.tensor(contexts), torch.tensor(negatives)
                targets, contexts, negatives = [], [], []

# --- 2. MODELS ---

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context, negatives):
        v_target = self.in_embed(target)                     
        v_context = self.out_embed(context)                  
        v_negatives = self.out_embed(negatives)              

        pos_score = torch.sum(v_target * v_context, dim=1)
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(v_negatives, v_target.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(pos_loss + neg_loss)

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, contexts, negatives):
        v_contexts = self.in_embed(contexts)                 
        v_context_avg = torch.mean(v_contexts, dim=1)        
        
        v_target = self.out_embed(target)                    
        v_negatives = self.out_embed(negatives)              

        pos_score = torch.sum(v_context_avg * v_target, dim=1)
        pos_loss = -F.logsigmoid(pos_score)

        neg_score = torch.bmm(v_negatives, v_context_avg.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(pos_loss + neg_loss)

# --- 3. EVALUATION & SEMANTIC ANALYSIS FUNCTIONS ---

def get_similar_words(model, word2idx, idx2word, target_word, top_k=5):
    """TASK 3.1: Calculates cosine similarity to find the nearest neighbors."""
    if target_word not in word2idx:
        return f"'{target_word}' is not in the vocabulary."
    
    embeddings = model.in_embed.weight.data
    target_idx = word2idx[target_word]
    target_vector = embeddings[target_idx]
    
    similarities = F.cosine_similarity(target_vector.unsqueeze(0), embeddings)
    top_scores, top_indices = torch.topk(similarities, top_k + 1)
    
    results = []
    for i in range(1, top_k + 1):  
        idx = top_indices[i].item()
        score = top_scores[i].item()
        results.append((idx2word[idx], round(score, 4)))
        
    return results

def get_analogy(model, word2idx, idx2word, word_a, word_b, word_c, top_k=1):
    """TASK 3.2: Solves analogies of the form: A is to B as C is to ?"""
    if word_a not in word2idx or word_b not in word2idx or word_c not in word2idx:
        return f"Missing vocabulary for analogy: {word_a}:{word_b} :: {word_c}:?"
    
    embeddings = model.in_embed.weight.data
    
    v_a = embeddings[word2idx[word_a]]
    v_b = embeddings[word2idx[word_b]]
    v_c = embeddings[word2idx[word_c]]
    
    target_vec = v_b - v_a + v_c
    
    similarities = F.cosine_similarity(target_vec.unsqueeze(0), embeddings)
    top_scores, top_indices = torch.topk(similarities, top_k + 4)
    
    results = []
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        word = idx2word[idx]
        
        if word not in [word_a, word_b, word_c]:
            score = top_scores[i].item()
            results.append((word, round(score, 4)))
            if len(results) == top_k:
                break
                
    return results

# --- 4. VISUALIZATION FUNCTION ---

def visualize_embeddings(model, word2idx, words_to_visualize, filename="embeddings_2d.png", method='tsne'):
    """TASK 4: Reduces word embeddings to 2D and plots them."""
    vectors = []
    valid_words = []
    
    for word in words_to_visualize:
        if word in word2idx:
            vectors.append(model.in_embed.weight.data[word2idx[word]].cpu().numpy())
            valid_words.append(word)
            
    if len(valid_words) < 5:
        print("Not enough valid words found in vocabulary to visualize.")
        return
        
    vectors = np.array(vectors)
    
    if method == 'tsne':
        perplexity = min(30, len(valid_words) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
        
    reduced_vectors = reducer.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    for i, word in enumerate(valid_words):
        x, y = reduced_vectors[i, 0], reduced_vectors[i, 1]
        plt.scatter(x, y, color='blue', alpha=0.6)
        plt.text(x + 0.02, y + 0.02, word, fontsize=10, alpha=0.8)
        
    plt.title(f"2D Word Embeddings Visualization ({method.upper()})", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

# --- 5. TRAINING PIPELINE ---

def train_word2vec(data_loader, model_type="skipgram", embed_dim=50, window_size=2, num_neg_samples=5, epochs=3, batch_size=512, lr=0.01):
    
    if model_type == "skipgram":
        model = SkipGramModel(data_loader.vocab_size, embed_dim)
        is_cbow = False
    elif model_type == "cbow":
        model = CBOWModel(data_loader.vocab_size, embed_dim)
        is_cbow = True
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        batches = data_loader.generate_batches(window_size, num_neg_samples, batch_size, is_cbow)
        
        for targets, contexts, negatives in batches:
            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        print(f"  Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / batch_count:.4f}")
        
    return model

if __name__ == "__main__":
    data_file = "corpus.txt"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run your preparation script to generate it.")
    else:
        print(f"Loading data from {data_file}...")
        with open(data_file, "r", encoding="utf-8") as f:
            # Read the entire text and split by whitespace to get the tokens
            text = f.read()
            real_tokens = text.split()
            
        # Initialize data loader
        data_loader = Word2VecData(real_tokens)
        
        # Define hyperparameter grid for the report
        model_types = ["skipgram", "cbow"]
        embed_dims = [50]
        window_sizes = [2, 4]
        neg_samples = [5]
        
        eval_words = ["research", "student", "phd", "exam"]
        
        words_to_plot = ["research", "student", "phd", "exam", "course", "faculty", "btech", "mtech", 
                         "ug", "pg", "institute", "technology", "science", "engineering", "computer",
                         "semester", "grade", "hostel", "fee", "admission", "project", "thesis"]
        
        analogies = [
            ("pg", "mtech", "phd"),           
            ("student", "course", "faculty"),
            ("science", "technology", "computer")
        ]

        print("\n" + "="*50)
        print("STARTING TRAINING AND EVALUATION")
        print("="*50)

        for m_type, dim, win, neg in itertools.product(model_types, embed_dims, window_sizes, neg_samples):
            print(f"\nTraining Model: {m_type.upper()} | Dim: {dim} | Window: {win} | Neg_Samples: {neg}")
            
            trained_model = train_word2vec(
                data_loader=data_loader,
                model_type=m_type, 
                embed_dim=dim, 
                window_size=win, 
                num_neg_samples=neg,
                epochs=5  
            )
            
            print("\n  Task 3.1: Top 5 Nearest Neighbors:")
            for word in eval_words:
                if word in data_loader.word2idx:
                    neighbors = get_similar_words(trained_model, data_loader.word2idx, data_loader.idx2word, word, top_k=5)
                    print(f"    '{word}' -> {neighbors}")
            
            print("\n  Task 3.2: Analogy Experiments:")
            for w_a, w_b, w_c in analogies:
                analogy_result = get_analogy(trained_model, data_loader.word2idx, data_loader.idx2word, w_a, w_b, w_c, top_k=3)
                print(f"    {w_a} : {w_b} :: {w_c} : ? -> {analogy_result}")
                
            print("\n  Task 4: Generating Visualization...")
            filename = f"visualization_{m_type}_w{win}.png"
            visualize_embeddings(trained_model, data_loader.word2idx, words_to_plot, filename=filename, method='tsne')