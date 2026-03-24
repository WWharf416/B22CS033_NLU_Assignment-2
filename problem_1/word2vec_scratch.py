import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import numpy as np
import json
import os
import itertools

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

# --- 3. EVALUATION FUNCTION ---

def get_similar_words(model, word2idx, idx2word, target_word, top_k=5):
    """Calculates cosine similarity to find the nearest neighbors of a word."""
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

# --- 4. TRAINING & EXPERIMENT PIPELINE ---

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
    data_file = "cleaned_tokens.json"
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run your preparation script to generate it.")
    else:
        print(f"Loading data from {data_file}...")
        with open(data_file, "r") as f:
            real_tokens = json.load(f)
            
        # Initialize data loader once to save time
        data_loader = Word2VecData(real_tokens)
        
        # Define hyperparameter grid for the report
        model_types = ["skipgram", "cbow"]
        embed_dims = [50]
        window_sizes = [2, 4]
        neg_samples = [5, 10]
        
        # Choose a few test words relevant to an academic corpus to evaluate semantic clustering
        test_words = ["student", "research", "course", "phd", "exam"]
        
        print("\n" + "="*50)
        print("STARTING HYPERPARAMETER EXPERIMENTS")
        print("="*50)

        # Loop through all combinations of hyperparameters
        for m_type, dim, win, neg in itertools.product(model_types, embed_dims, window_sizes, neg_samples):
            print(f"\nTraining Model: {m_type.upper()} | Dim: {dim} | Window: {win} | Neg_Samples: {neg}")
            
            # Train the model
            trained_model = train_word2vec(
                data_loader=data_loader,
                model_type=m_type, 
                embed_dim=dim, 
                window_size=win, 
                num_neg_samples=neg,
                epochs=3  # Keep epochs low while testing the grid, increase for final run
            )
            
            # Evaluate using Cosine Similarity immediately after training
            print("  Evaluation (Nearest Neighbors):")
            for word in test_words:
                if word in data_loader.word2idx:
                    neighbors = get_similar_words(trained_model, data_loader.word2idx, data_loader.idx2word, word, top_k=3)
                    print(f"    '{word}' -> {neighbors}")