import os
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

# =========================
# Config
# =========================
SEQ_LEN = 30
N_GEN = 1000
SCORE_BATCH = 16
MAX_LEN_T5 = 100
OUT_CSV = "results/evodiff.csv"

TOX_CKPT = "best_toxic_model.pt"
AMP_CKPT = "head_only.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Classifier Head
# =========================
class ProteinHead(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=2, nhead=4, nlayers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=7, padding=3)

        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.classifier = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(150, num_classes),
        )

    def forward(self, hidden_states):
        x = hidden_states.transpose(1, 2)  # [B, 1024, L]
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)  # [B, 300, L]
        x = x.transpose(1, 2)               # [B, L, 300]
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def load_heads():
    shared_encoder = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc"
    ).to(device).eval()
    prot_tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )

    toxic_model = ProteinHead(num_classes=2).to(device).eval()
    amp_model = ProteinHead(num_classes=2).to(device).eval()

    tox_ckpt = torch.load(TOX_CKPT, map_location=device)

    if isinstance(tox_ckpt, dict) and "conv1" in tox_ckpt:
        toxic_model.conv1.load_state_dict(tox_ckpt["conv1"])
        toxic_model.conv2.load_state_dict(tox_ckpt["conv2"])
        toxic_model.conv3.load_state_dict(tox_ckpt["conv3"])
        toxic_model.transformer.load_state_dict(tox_ckpt["transformer"])
        toxic_model.classifier.load_state_dict(tox_ckpt["classifier"])
    else:
        toxic_model.load_state_dict(tox_ckpt)

    amp_ckpt = torch.load(AMP_CKPT, map_location=device)
    if isinstance(amp_ckpt, dict) and "conv1" in amp_ckpt:
        amp_model.conv1.load_state_dict(amp_ckpt["conv1"])
        amp_model.conv2.load_state_dict(amp_ckpt["conv2"])
        amp_model.conv3.load_state_dict(amp_ckpt["conv3"])
        amp_model.transformer.load_state_dict(amp_ckpt["transformer"])
        amp_model.classifier.load_state_dict(amp_ckpt["classifier"])
    else:
        amp_model.load_state_dict(amp_ckpt)

    return shared_encoder, prot_tokenizer, amp_model, toxic_model


@torch.no_grad()
def score_sequences(seqs, shared_encoder, prot_tokenizer, amp_model, toxic_model, batch_size=SCORE_BATCH):
    pamps, ptoxs = [], []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        batch_spaced = [" ".join(list(s)) for s in batch]
        enc = prot_tokenizer(
            batch_spaced,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN_T5,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            h = shared_encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            amp_logits = amp_model(h)
            tox_logits = toxic_model(h)

            amp_probs = torch.softmax(amp_logits, dim=-1)[:, 1].detach().cpu().tolist()
            tox_probs = torch.softmax(tox_logits, dim=-1)[:, 1].detach().cpu().tolist()

        pamps.extend(amp_probs)
        ptoxs.extend(tox_probs)
    return pamps, ptoxs


def generate_n(model, tokenizer, n=N_GEN, seq_len=SEQ_LEN, batch_size=200):
    seqs = []
    model.eval()
    while len(seqs) < n:
        _, s = generate_oaardm(model, tokenizer, seq_len=seq_len, batch_size=batch_size, device=device)
        seqs.extend(s)
    return seqs[:n]


def save_csv(path, seqs, pamps, ptoxs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "pAMP", "pTox"])
        for s, a, t in zip(seqs, pamps, ptoxs):
            w.writerow([s, f"{a:.6f}", f"{t:.6f}"])


def main():
    model, collater, tokenizer, scheme = OA_DM_38M()
    model = model.to(device).eval()

    shared_encoder, prot_tokenizer, amp_model, toxic_model = load_heads()

    seqs = generate_n(model, tokenizer, n=N_GEN, seq_len=SEQ_LEN)
    pamps, ptoxs = score_sequences(seqs, shared_encoder, prot_tokenizer, amp_model, toxic_model)

    save_csv(OUT_CSV, seqs, pamps, ptoxs)
    print(f"[OK] Saved {OUT_CSV} ({len(seqs)} seqs)")


if __name__ == "__main__":
    main()
