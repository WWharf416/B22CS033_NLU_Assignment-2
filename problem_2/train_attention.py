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

# Train/Validation Split (90/10)
random.seed(42)
random.shuffle(names)
split_idx = int(len(names) * 0.9)
train_names = names[:split_idx]
val_names = names[split_idx:]

# --- Model ---
class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, trg=None, max_len=15):
        batch_size = src.shape[0]
        embedded_src = self.embedding(src)
        enc_outputs, hidden = self.encoder(embedded_src)
        
        if trg is not None:
            trg_len = trg.shape[1]
            outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
            dec_input = trg[:, 0].unsqueeze(1)
            
            for t in range(1, trg_len):
                embedded_dec = self.embedding(dec_input)
                hidden_rep = hidden.permute(1, 0, 2).repeat(1, enc_outputs.size(1), 1)
                energy = torch.tanh(self.attn(torch.cat((hidden_rep, enc_outputs), dim=2)))
                attention = F.softmax(self.v(energy).squeeze(2), dim=1).unsqueeze(1)
                context = torch.bmm(attention, enc_outputs)
                
                rnn_input = torch.cat((embedded_dec, context), dim=2)
                out, hidden = self.decoder(rnn_input, hidden)
                prediction = self.fc(out.squeeze(1))
                outputs[:, t, :] = prediction
                dec_input = trg[:, t].unsqueeze(1)
            return outputs
        else:
            outputs = []
            dec_input = torch.tensor([[char_to_idx['<SOS>']]], device=src.device)
            for _ in range(max_len):
                embedded_dec = self.embedding(dec_input)
                hidden_rep = hidden.permute(1, 0, 2).repeat(1, enc_outputs.size(1), 1)
                energy = torch.tanh(self.attn(torch.cat((hidden_rep, enc_outputs), dim=2)))
                attention = F.softmax(self.v(energy).squeeze(2), dim=1).unsqueeze(1)
                context = torch.bmm(attention, enc_outputs)
                
                rnn_input = torch.cat((embedded_dec, context), dim=2)
                out, hidden = self.decoder(rnn_input, hidden)
                prediction = self.fc(out.squeeze(1))
                
                probs = F.softmax(prediction, dim=-1)
                next_char = torch.multinomial(probs, 1).item()
                if next_char == char_to_idx['<EOS>']: break
                
                outputs.append(idx_to_char[next_char])
                dec_input = torch.tensor([[next_char]], device=src.device)
            return "".join(outputs)

# --- Training Loop with Early Stopping ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionRNN(vocab_size, 64, 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Lowered LR for stability
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])

MAX_EPOCHS = 100
PATIENCE = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print(f"Training Attention RNN on {device} with Early Stopping (Patience: {PATIENCE})...")

for epoch in range(MAX_EPOCHS):
    # Training Phase
    model.train()
    total_train_loss = 0
    random.shuffle(train_names)
    for name in train_names:
        seq = [char_to_idx['<SOS>']] + [char_to_idx[c] for c in name] + [char_to_idx['<EOS>']]
        src = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
        trg = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        outputs = model(src, trg)
        loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), trg[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_names)

    # Validation Phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for name in val_names:
            seq = [char_to_idx['<SOS>']] + [char_to_idx[c] for c in name] + [char_to_idx['<EOS>']]
            src = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            trg = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            
            outputs = model(src, trg)
            loss = criterion(outputs[:, 1:].reshape(-1, vocab_size), trg[:, 1:].reshape(-1))
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_names)
    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy() # Save the best weights
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
        
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at Epoch {epoch+1}!")
        break

# Load the best weights back into the model before generation
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# --- Generation ---
print("\nGenerating 1000 names...")
model.eval()
generated_names = []
with torch.no_grad():
    for _ in range(1000):
        seed_char = random.choice(chars)
        src = torch.tensor([[char_to_idx['<SOS>'], char_to_idx[seed_char]]], device=device)
        gen_name = seed_char + model(src)
        generated_names.append(gen_name)

with open("generated_attn.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(generated_names))
print("Saved to generated_attn.txt")