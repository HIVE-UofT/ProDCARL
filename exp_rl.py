import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

# =========================
# Config
# =========================
SEQ_LEN = 30
RL_STEPS = 100
RL_BATCH = 200
TOPK_FRAC = 0.30
ENTROPY_LAMBDA = 3.0
LR_RL = 2e-4

N_GEN = 1000
SCORE_BATCH = 16
MAX_LEN_T5 = 100
OUT_CSV = "results/evodiff_rl.csv"

TOX_CKPT = "best_toxic_model.pt"
AMP_CKPT = "head_only.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        x = hidden_states.transpose(1, 2)
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.transpose(1, 2)
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

        h = shared_encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        amp_logits = amp_model(h)
        tox_logits = toxic_model(h)

        amp_probs = torch.softmax(amp_logits, dim=-1)[:, 1].detach().cpu().tolist()
        tox_probs = torch.softmax(tox_logits, dim=-1)[:, 1].detach().cpu().tolist()

        pamps.extend(amp_probs)
        ptoxs.extend(tox_probs)
    return pamps, ptoxs


def sample_sequences(model, tokenizer, batch_size=RL_BATCH, seq_len=SEQ_LEN):
    _, seqs = generate_oaardm(model, tokenizer, seq_len=seq_len, batch_size=batch_size, device=device)
    return seqs


def rl_train(model, tokenizer, shared_encoder, prot_tokenizer, amp_model, toxic_model):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_RL)

    for step in range(RL_STEPS):
        seqs = sample_sequences(model, tokenizer, batch_size=RL_BATCH, seq_len=SEQ_LEN)

        pamps, ptoxs = score_sequences(seqs, shared_encoder, prot_tokenizer, amp_model, toxic_model, batch_size=SCORE_BATCH)
        rewards = torch.tensor([a + (1.0 - t) for a, t in zip(pamps, ptoxs)], device=device)
        rewards = rewards.pow(2)

        # Compute surrogate log pi at t=0
        tokenized = [tokenizer.tokenize(s) for s in seqs]
        max_len_batch = max(len(t) for t in tokenized)
        token_ids = torch.zeros(RL_BATCH, max_len_batch, dtype=torch.long, device=device)
        for i, t in enumerate(tokenized):
            token_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        timesteps = torch.zeros(RL_BATCH, dtype=torch.long, device=device)  # t=0 (final step)
        logits = model(token_ids, timesteps)  # [B, L, V]
        log_probs = F.log_softmax(logits, dim=-1)

        seq_logp = []
        for i in range(RL_BATCH):
            L = len(tokenized[i])
            tok = token_ids[i, :L]
            lp = log_probs[i, :L].gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            seq_logp.append(lp.mean())
        seq_logp = torch.stack(seq_logp)

        topk = int(RL_BATCH * TOPK_FRAC)
        _, idx = rewards.topk(topk)

        rl_loss = -(seq_logp[idx] * (rewards[idx] - rewards.mean())).mean()

        probs = log_probs.exp()
        entropy = -(probs * (probs + 1e-12).log()).sum(-1).mean()

        loss = rl_loss - ENTROPY_LAMBDA * entropy

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"[RL] step={step:03d} avgR={rewards.mean().item():.3f} loss={loss.item():.3f} ent={entropy.item():.3f}")


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
    model = model.to(device)

    shared_encoder, prot_tokenizer, amp_model, toxic_model = load_heads()

    # RL train
    rl_train(model, tokenizer, shared_encoder, prot_tokenizer, amp_model, toxic_model)

    # Generate + score + save
    seqs = generate_n(model, tokenizer, n=N_GEN, seq_len=SEQ_LEN)
    pamps, ptoxs = score_sequences(seqs, shared_encoder, prot_tokenizer, amp_model, toxic_model)

    save_csv(OUT_CSV, seqs, pamps, ptoxs)
    print(f"[OK] Saved {OUT_CSV} ({len(seqs)} seqs)")


if __name__ == "__main__":
    main()
