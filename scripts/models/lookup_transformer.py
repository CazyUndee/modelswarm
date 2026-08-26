"""Lookup-transformer for tabular data: exact-value embeddings + transformer.

Each feature value is treated as a discrete token (exact string). Learnable
embeddings per feature, then transformer self-attention over feature tokens.
Fold-safe: vocab built from training split only.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def _value_to_token(v):
    if pd.isna(v):
        return "__MISSING__"
    # Preserve exact representation for numerics (2 decimals matches 0.01 grid)
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.2f}"
    return str(v)


def build_vocab(df, columns):
    vocabs = {}
    for col in columns:
        tokens = df[col].apply(_value_to_token).unique()
        # Reserve 0 for unknown
        mapping = {tok: i + 1 for i, tok in enumerate(tokens)}
        mapping["__UNKNOWN__"] = 0
        vocabs[col] = mapping
    return vocabs


def encode_df(df, columns, vocabs):
    arr = np.zeros((len(df), len(columns)), dtype=np.int64)
    for j, col in enumerate(columns):
        mapping = vocabs[col]
        tokens = df[col].apply(_value_to_token)
        arr[:, j] = tokens.map(lambda t: mapping.get(t, 0)).values
    return arr


class LookupTransformer(nn.Module):
    def __init__(self, vocab_sizes, d_model=32, nhead=4, num_layers=2, dropout=0.1, n_features=15):
        super().__init__()
        self.n_features = n_features
        # Per-feature embedding tables (different vocab sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_sizes[i] + 1, d_model, padding_idx=0)
            for i in range(n_features)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (batch, n_features) int tokens
        embs = torch.stack([self.embeddings[i](x[:, i]) for i in range(self.n_features)], dim=1)
        # (batch, n_features, d_model)
        out = self.transformer(embs)  # (batch, n_features, d_model)
        pooled = out.mean(dim=1)  # (batch, d_model)
        return self.head(pooled).squeeze(-1)


def train_lookup_transformer(X_train, y_train, X_val, y_val, columns, params, device="cpu"):
    d_model = int(params.get("d_model", 32))
    nhead = int(params.get("nhead", 4))
    num_layers = int(params.get("num_layers", 2))
    dropout = float(params.get("dropout", 0.1))
    lr = float(params.get("lr", 1e-3))
    batch_size = int(params.get("batch_size", 1024))
    epochs = int(params.get("epochs", 30))
    patience = int(params.get("patience", 5))

    vocabs = build_vocab(X_train[columns], columns)
    vocab_sizes = [len(vocabs[c]) for c in columns]

    Xtr_enc = encode_df(X_train, columns, vocabs)
    Xva_enc = encode_df(X_val, columns, vocabs)

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr_enc), torch.from_numpy(y_train.values.astype(np.float32))
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xva_enc), torch.from_numpy(y_val.values.astype(np.float32))
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model = LookupTransformer(vocab_sizes, d_model=d_model, nhead=nhead,
                              num_layers=num_layers, dropout=dropout, n_features=len(columns))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                preds.append(torch.sigmoid(model(xb)).cpu().numpy())
        preds = np.concatenate(preds)
        auc = roc_auc_score(y_val.values, preds)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Store vocab for test-time encoding
    model.vocabs = vocabs
    model.columns = columns
    return model


def predict_lookup_transformer(model, X, device="cpu"):
    model.eval()
    columns = model.columns
    vocabs = model.vocabs
    X_enc = encode_df(X, columns, vocabs)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_enc))
    loader = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)
