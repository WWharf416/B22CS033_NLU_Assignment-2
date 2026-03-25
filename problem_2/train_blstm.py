import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# --- Data Prep ---
with open("names.txt", "r", encoding="utf-8") as f:
    names = [line.strip().lower() for line in f if line.strip()]

chars = sorted(list(set(''.join(names))))
vocab = ['<PAD>', '<SOS>', '<EOS>'] + chars
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)

random.seed(42)
random.shuffle(names)
split_idx = int(len(names) * 0.9)
train_names = names[:split_idx]
val_names = names[split_idx:]

# --- Model ---
class Seq2SeqBLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size * 2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, src, trg=None):
        embedded_src = self.embedding(src)
        _, (h_n, c_n) = self.encoder(embedded_src)
        
        h_n_concat = torch.cat((h_n[0:1], h_n[1:2]), dim=2)
        c_n_concat = torch.cat((c_n[0:1], c_n[1:2]), dim=2)
        hidden = (h_n_concat, c_n_concat)
        
        if trg is None:
            return hidden 
            
        embedded_trg = self.embedding(trg)
        out, _ = self.decoder(embedded_trg, hidden)
        logits = self.fc(out)
        return logits

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqBLSTM(vocab_size, 64, 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])

MAX_EPOCHS = 30
PATIENCE = 5
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print(f"Training Seq2Seq BLSTM on {device}...")

for epoch in range(MAX_EPOCHS):
    model.train()
    total_train_loss = 0
    random.shuffle(train_names)
    
    for name in train_names:
        if len(name) < 2: continue 
        seq = [char_to_idx['<SOS>']] + [char_to_idx[c] for c in name] + [char_to_idx['<EOS>']]
        
        # Split: Encoder sees <SOS> + First Char. Decoder predicts the rest.
        src = torch.tensor(seq[:2], dtype=torch.long).unsqueeze(0).to(device)
        trg_in = torch.tensor(seq[1:-1], dtype=torch.long).unsqueeze(0).to(device)
        trg_out = torch.tensor(seq[2:], dtype=torch.long).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        logits = model(src, trg_in)
        loss = criterion(logits.view(-1, vocab_size), trg_out.view(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_names)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for name in val_names:
            if len(name) < 2: continue
            seq = [char_to_idx['<SOS>']] + [char_to_idx[c] for c in name] + [char_to_idx['<EOS>']]
            src = torch.tensor(seq[:2], dtype=torch.long).unsqueeze(0).to(device)
            trg_in = torch.tensor(seq[1:-1], dtype=torch.long).unsqueeze(0).to(device)
            trg_out = torch.tensor(seq[2:], dtype=torch.long).unsqueeze(0).to(device)
            
            logits = model(src, trg_in)
            loss = criterion(logits.view(-1, vocab_size), trg_out.view(-1))
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_names)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        print("Early stopping triggered!")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

# --- Generation ---
print("Generating 1000 names...")
model.eval()
generated_names = []
with torch.no_grad():
    for _ in range(1000):
        seed_char = random.choice(chars)
        seed_seq = torch.tensor([[char_to_idx['<SOS>'], char_to_idx[seed_char]]], device=device)
        
        hidden = model(seed_seq)
        current_char = torch.tensor([[char_to_idx[seed_char]]], device=device)
        name_chars = [seed_char]
        
        for _ in range(14):
            embedded = model.embedding(current_char)
            out, hidden = model.decoder(embedded, hidden)
            logits = model.fc(out)
            
            probs = F.softmax(logits[:, -1, :], dim=-1).squeeze()
            next_char_idx = torch.multinomial(probs, 1).item()
            if next_char_idx == char_to_idx['<EOS>']: break
            
            name_chars.append(idx_to_char[next_char_idx])
            current_char = torch.tensor([[next_char_idx]], device=device)
            
        generated_names.append(''.join(name_chars))

with open("generated_blstm.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(generated_names))
print("Saved to generated_blstm.txt")