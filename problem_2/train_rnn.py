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

# --- Model ---
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VanillaRNN(vocab_size, 64, 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])

print(f"Training Vanilla RNN on {device}...")
for epoch in range(15):
    total_loss = 0
    random.shuffle(names)
    for name in names:
        seq = [char_to_idx['<SOS>']] + [char_to_idx[c] for c in name] + [char_to_idx['<EOS>']]
        x = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(seq[1:], dtype=torch.long).unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(names):.4f}")

# --- Generation ---
print("Generating 1000 names...")
model.eval()
generated_names = []
with torch.no_grad():
    for _ in range(1000):
        current_char = torch.tensor([[char_to_idx['<SOS>']]], device=device)
        hidden = None
        name_chars = []
        for _ in range(15):
            logits, hidden = model(current_char, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1).squeeze()
            next_char_idx = torch.multinomial(probs, 1).item()
            if next_char_idx == char_to_idx['<EOS>']: break
            name_chars.append(idx_to_char[next_char_idx])
            current_char = torch.tensor([[next_char_idx]], device=device)
        generated_names.append(''.join(name_chars))

with open("generated_rnn.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(generated_names))
print("Saved to generated_rnn.txt")