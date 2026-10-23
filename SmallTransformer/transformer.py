import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
#from google.colab import files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

df = pd.read_csv("Dataset/sentiment_dataset.csv")
#print(df.head())

def tokenize(text):
	return text.lower().split()

vocab = sorted( set(word for sentence in df["review"] for word in tokenize(sentence)) )

# print(vocab)
# print(type(vocab))

word2idx = {word: i+4 for i, word in enumerate(vocab)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1
word2idx["<CLS>"] = 2
word2idx["<EOS>"] = 3

# print(word2idx)
idx2word = {v: k for k, v in word2idx.items()}
#print(idx2word)
vocab_size = len(word2idx)
#print(vocab_size)

def encode(sentence, max_len=15):
    tokens = [word2idx.get(w, 1) for w in tokenize(sentence)]
    tokens = [2] + tokens[:max_len-2] + [3]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens
#data = [encode(s, 15) for s in df["review"]]
#print(data)
X = torch.tensor([encode(s, 15) for s in df["review"]]).to(device)
label2idx = {"negative": 0, "neutral": 1, "positive": 2}
classes = ["negative", "neutral", "positive"]
y = torch.tensor([label2idx[s] for s in df["sentiment"]]).to(device)

## Personal Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2*i/d_model)))
                if i + 1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/d_model)))
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


## Mini Transformer Encoder
class SelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(context)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = SelfAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model*4), nn.ReLU(), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ff(x))

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=64, heads=4, max_len=15, num_classes=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(d_model, heads)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)
        x = self.encoder(x)
        cls_token = x[:, 0]  # [CLS]
        return self.classifier(cls_token)

## Train the Model

model = SentimentClassifier(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(300):
    model.train()
    logits = model(X)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 30 == 0:
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean()
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Accuracy={acc:.2f}")

## Sentiment Predictions

def predict_sentiment(sentence):
    model.eval()
    with torch.no_grad():
        encoded = torch.tensor([encode(sentence)]).to(device)
        logits = model(encoded)
        pred = torch.argmax(logits, dim=1).item()
        return classes[pred]

## Inference Function

while True:
    review = input("\nEnter a review (or 'exit'): ")
    if review.lower() == 'exit':
        break
    print("Predicted Sentiment:", predict_sentiment(review))
