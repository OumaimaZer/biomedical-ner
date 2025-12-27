import torch
import torch.nn as nn
from typing import Dict, Optional, List
from torchcrf import CRF
import numpy as np

# ==================== CharCNN ====================
class CharCNN(nn.Module):
    def __init__(self, char_vocab_size: int, char_embed_dim: int = 50,
                 num_filters: int = 32, kernel_sizes: List[int] = [3,5,7], dropout: float = 0.5):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_embed_dim, out_channels=num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        B, T, W = char_ids.size()
        char_ids = char_ids.view(B*T, W)
        emb = self.char_embedding(char_ids).transpose(1,2)
        conv_feats = []
        for conv in self.convs:
            x = torch.relu(conv(emb))
            x, _ = torch.max(x, dim=2)
            conv_feats.append(x)
        out = torch.cat(conv_feats, dim=1)
        return self.dropout(out).view(B, T, -1)

# ==================== CharBiLSTM ====================
class CharBiLSTM(nn.Module):
    def __init__(self, char_vocab_size: int, char_embed_dim: int = 50,
                 hidden_dim: int = 50, dropout: float = 0.5):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(char_embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        B, T, W = char_ids.size()
        char_ids = char_ids.view(B*T, W)
        emb = self.char_embedding(char_ids)
        _, (h, _) = self.lstm(emb)
        out = torch.cat([h[0], h[1]], dim=1)
        return self.dropout(out).view(B, T, -1)

# ==================== Manhattan Attention ====================
class ManhattanAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        B, T, D = h.shape
        hi = h.unsqueeze(2).expand(B, T, T, D)
        hj = h.unsqueeze(1).expand(B, T, T, D)
        dist = torch.abs(hi - hj).sum(-1)                # Manhattan distance
        score = -self.W(hj).squeeze(-1) * dist
        score = score.masked_fill(~mask.unsqueeze(1), -1e9)
        alpha = torch.softmax(score, -1)
        ctx = torch.matmul(alpha, h)
        return torch.cat([h, ctx], -1)                  # concat context

# ==================== CombinatorialNER ====================
class CombinatorialNER(nn.Module):
    def __init__(self, vocab_size: int, char_vocab_size: int, tag_to_idx: Dict[str,int],
                 dataset: str = "JNLPBA", use_char_cnn=True, use_char_lstm=True,
                 use_attention=True, use_fc_fusion=True, pretrained_embeddings: Optional[np.ndarray] = None,
                 word_embed_dim: int = 200, lstm_hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()

        # Kernel sizes by dataset
        if dataset == "JNLPBA":
            cnn_kernels = [3,5,7]
        elif dataset == "NCBI":
            cnn_kernels = [2,3,4]
        else:
            cnn_kernels = [3,5,7]

        self.use_char_cnn = use_char_cnn
        self.use_char_lstm = use_char_lstm
        self.use_attention = use_attention
        self.use_fc_fusion = use_fc_fusion

        # Word embeddings
        if pretrained_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float), padding_idx=0, freeze=False
            )
        else:
            self.word_embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)

        # Char encoders
        if self.use_char_cnn:
            self.char_cnn = CharCNN(char_vocab_size, char_embed_dim=50,
                                    num_filters=32, kernel_sizes=cnn_kernels, dropout=dropout)
        if self.use_char_lstm:
            self.char_lstm = CharBiLSTM(char_vocab_size, char_embed_dim=50, hidden_dim=50, dropout=dropout)

        # Combined dimension
        char_dim = 0
        if self.use_char_cnn:
            char_dim += 32*len(cnn_kernels)
        if self.use_char_lstm:
            char_dim += 100  # 50*2

        combined_dim = word_embed_dim + char_dim

        # FC fusion
        # ==================== FC fusion ====================
        if self.use_fc_fusion:
            if dataset == "NCBI":
                self.fusion = nn.Sequential(
                    nn.Linear(combined_dim, 200),
                    nn.ReLU(),              # ReLU only for NCBI
                    nn.Dropout(dropout)
                )
            else:  # JNLPBA or others
                self.fusion = nn.Sequential(
                    nn.Linear(combined_dim, 200),
                    nn.Dropout(dropout)     # no activation
                )
            lstm_input_dim = 200
        else:
            lstm_input_dim = combined_dim


        # Context BiLSTM
        self.context_lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim//2, batch_first=True, bidirectional=True)
        if self.use_attention:
            self.attention_layer = ManhattanAttention(lstm_hidden_dim)

        # Emission & CRF
        self.emission = nn.Linear(lstm_hidden_dim, len(tag_to_idx))
        self.crf = CRF(len(tag_to_idx))

    def forward(self, word_ids, char_ids, mask, tags=None):
        word_emb = self.word_embedding(word_ids)
        char_embs = []
        if self.use_char_cnn:
            char_embs.append(self.char_cnn(char_ids))
        if self.use_char_lstm:
            char_embs.append(self.char_lstm(char_ids))
        combined = torch.cat([word_emb] + char_embs, dim=-1)
        if self.use_fc_fusion:
            combined = self.fusion(combined)
        lstm_out, _ = self.context_lstm(combined)
        if self.use_attention:
            lstm_out = self.attention_layer(lstm_out, mask)
        emissions = self.emission(lstm_out).transpose(0,1)
        mask = mask.transpose(0,1)
        if tags is not None:
            tags = tags.transpose(0,1)
            return -self.crf(emissions, tags, mask=mask).mean()
        else:
            return self.crf.decode(emissions, mask=mask)
