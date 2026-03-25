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
class Seq2SeqAttentionRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.RNN(embed_size + hidden_size, hidden_size, batch_first=True)
        
        # Basic Bahdanau Attention
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, trg_in=None, max_len=15):
        batch_size = src.shape[0]
        embedded_src = self.embedding(src)
        enc_outputs, hidden = self.encoder(embedded_src)
        
        # If generating (trg_in is None)
        if trg_in is None:
            outputs = []
            dec_input = src[:, -1].unsqueeze(1) # Start with seed_char
            
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
            return outputs

        # If training
        trg_len = trg_in.shape[1]
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        dec_input = trg_in[:, 0].unsqueeze(1)
        
        for t in range(trg_len):
            embedded_dec = self.embedding(dec_input)
            hidden_rep = hidden.permute(1, 0, 2).repeat(1, enc_outputs.size(1), 1)
            
            energy = torch.tanh(self.attn(torch.cat((hidden_rep, enc_outputs), dim=2)))
            attention = F.softmax(self.v(energy).squeeze(2), dim=1).unsqueeze(1)
            context = torch.bmm(attention, enc_outputs)
            
            rnn_input = torch.cat((embedded_dec, context), dim=2)
            out, hidden = self.decoder(rnn_input, hidden)
            
            prediction = self.fc(out.squeeze(1))
            outputs[:, t, :] = prediction
            
            if t < trg_len - 1:
                dec_input = trg_in[:, t+1].unsqueeze(1)
                
        return outputs

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqAttentionRNN(vocab_size, 64, 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])

MAX_EPOCHS = 30
PATIENCE = 5
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print(f"Training Attention RNN on {device}...")

for epoch in range(MAX_EPOCHS):
    model.train()
    total_train_loss = 0
    random.shuffle(train_names)
    
    for name in train_names:
        if len(name) < 2: continue
        seq = [char_to_idx['<SOS>']] + [char_to_idx[c] for c in name] + [char_to_idx['<EOS>']]
        
        src = torch.tensor(seq[:2], dtype=torch.long).unsqueeze(0).to(device)
        trg_in = torch.tensor(seq[1:-1], dtype=torch.long).unsqueeze(0).to(device)
        trg_out = torch.tensor(seq[2:], dtype=torch.long).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        outputs = model(src, trg_in)
        loss = criterion(outputs.view(-1, vocab_size), trg_out.view(-1))
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
            
            outputs = model(src, trg_in)
            loss = criterion(outputs.view(-1, vocab_size), trg_out.view(-1))
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
        src = torch.tensor([[char_to_idx['<SOS>'], char_to_idx[seed_char]]], device=device)
        
        # Generation is now handled cleanly inside the forward method
        generated_chars = model(src)
        generated_names.append(seed_char + ''.join(generated_chars))

with open("generated_attn.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(generated_names))
print("Saved to generated_attn.txt")