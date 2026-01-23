import tqdm
from evodiff.losses import OAMaskedCrossEntropyLoss, D3PMCELoss, D3PMLVBLoss
from evodiff.pretrained import OA_DM_38M
from evodiff.pretrained import OA_DM_640M
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset
from Bio import SeqIO
import random
from evodiff.generate import generate_oaardm
from sequence_models.samplers import SortishSampler, ApproxBatchSampler
import os
import csv

class MinimalFastaDataset(Dataset):
    def __init__(self, fasta_file, max_len=512):
        self.sequences = []
        with open(fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)
                if len(seq) > 10:  
                    self.sequences.append(seq)
        self.max_len = max_len
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            start = random.randint(0, len(seq) - self.max_len)
            seq = seq[start:start + self.max_len]
        return seq

class ProteinHead(nn.Module):
    def __init__(self, embed_dim=1024, num_classes=2, hidden_dim=300, nhead=4, nlayers=2):
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
            nn.Linear(150, num_classes)
        )
    def forward(self, hidden_states):
        x = hidden_states.transpose(1, 2)
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)   # [B, 300, L]
        x = x.transpose(1, 2)                # [B, L, 300]
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)

def predict(seq, max_len=100):
    seq = " ".join(list(seq))
    encoding = prot_tokenizer(seq, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        toxic_logits = toxic_model(hidden_states)
        amp_logits= amp_model(hidden_states)

    toxic_probs = torch.softmax(toxic_logits, dim=-1).cpu()
    amp_probs= torch.softmax(amp_logits, dim=-1).cpu()

    amp_score   = float(amp_probs[0][1])
    toxic_score = float(toxic_probs[0][1])
    return {
        "toxic_probs": toxic_probs.numpy(),
        "amp_probs": amp_probs.numpy()
    }

def compute_reward(seq):
    result = predict(seq)
    amp_score   = float(result["amp_probs"][0][1])
    toxic_score = float(result["toxic_probs"][0][1])
    return torch.tensor(amp_score + (1 - toxic_score), device=device)


def sample_sequences(model, tokenizer, num_seqs=200, max_len=30, device="cuda"):
    samples, seqs = generate_oaardm(
        model, tokenizer, seq_len=max_len,batch_size=400, device=device
    )
    return seqs


def train_rl(num_steps=50, batch_size=200, max_len=30,amp_seqs=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for step in range(num_steps):
        seqs = sample_sequences(model, tokenizer, num_seqs=200, max_len=max_len)
        rewards = torch.stack([compute_reward(s) for s in seqs])
        rewards = torch.pow(rewards, 2)
        tokenized = [tokenizer.tokenize(s) for s in seqs]
        B = len(seqs)
        max_len_batch = max(len(t) for t in tokenized)
        token_ids = torch.zeros(B, max_len_batch, dtype=torch.long).to(device)
        for i, t in enumerate(tokenized):
            token_ids[i, :len(t)] = torch.tensor(t)

        timesteps = torch.tensor([0] * B).to(device)

        # (4) Forward
        logits = model(token_ids, timesteps)  # [B, L, V]
        log_probs = F.log_softmax(logits, dim=-1)

        seq_log_probs = []
        for i in range(batch_size):
            length = len(tokenized[i])
            tok = token_ids[i, :length]
            logp = log_probs[i, :length].gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            seq_log_probs.append(logp.mean())
        seq_log_probs = torch.stack(seq_log_probs)

        topk = int(len(rewards) * 0.30)
        _, idx = rewards.topk(topk)
        rl_loss = -(seq_log_probs[idx].sum(-1) * (rewards[idx] - rewards.mean())).mean()
        probs = log_probs.exp()
        entropy = -(probs * probs.log()).sum(-1).mean()
        loss = rl_loss - 3 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"Step {step} | AvgReward={rewards.mean().item():.3f} | Loss={loss.item():.3f}")

def load_fasta(filepath):
    seqs = []
    with open(filepath, "r") as f:
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    seqs.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            seqs.append("".join(current_seq))
    return seqs


def warmup_supervised(model, tokenizer, amp_seqs, epochs=3, max_len=50):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        for i,seq in enumerate(amp_seqs):
            # 1. Tokenize
            print(f"seq:{seq}")
            tokens = tokenizer.tokenize(" ".join(list(seq)))
            print(f"tokens:{tokens}")
            ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            print(f"ids:{ids}")
            timesteps = torch.zeros(1, dtype=torch.long).to(device)
            print(f"[DEBUG] Seq {i}: len={len(tokens)}, timestep={timesteps.item()}")
            # 2. Forward
            logits = model(ids, timesteps)  # [1, L, V]
            print(f"[DEBUG] logits.shape = {logits.shape}, ids.shape = {ids.shape}")
            # 3. Cross entropy loss
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
            )
            # loss_func = OAMaskedCrossEntropyLoss(reweight=True)
            print(f"[DEBUG] loss = {loss.item():.4f}")
            # 4. Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


def fine_tune_oadm(model, train_dataset, tokenizer, collater,
                   fine_tune_config, device='cuda'):

    train_loader = DataLoader(
        train_dataset,
        batch_size=fine_tune_config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collater
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fine_tune_config['lr'],
        weight_decay=fine_tune_config.get('weight_decay', 0.0)
    )

    loss_func = OAMaskedCrossEntropyLoss(reweight=True)

  
    for epoch in range(fine_tune_config['epochs']):
        total_loss = 0
        total_acc = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(train_loader):
            src, timestep, tgt, mask = batch
            src = src.to(device)
            timestep = timestep.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            input_mask = (src != tokenizer.pad_id).float()

         
            optimizer.zero_grad()
            outputs = model(src, timestep, input_mask=input_mask.unsqueeze(-1))

            ce_loss, nll_loss = loss_func(outputs, tgt, mask, timestep, input_mask)
            #accu = accu_func(outputs, tgt, mask) * mask.sum()
            ce_loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            #total_acc += accu.item()
            total_tokens += mask.sum().item()

            if batch_idx % 1 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = total_acc / total_tokens
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}')

        if epoch % fine_tune_config['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader)
            }, f'finetune_checkpoint_epoch_{epoch}.pth')

@torch.no_grad()
def generate_and_save(model, tokenizer, enc, tok, amp, tox):
    seqs = []
    while len(seqs) < 1000:
        _, s = generate_oaardm(
            model,
            tokenizer,
            seq_len=30,
            batch_size=200,
            device=device
        )
        seqs += s
    seqs = seqs[:1000]

    # ===== compute pAMP / pTox =====
    batch = [" ".join(list(s)) for s in seqs]
    enc_in = tok(
        batch,
        padding="max_length",
        truncation=True,
        max_length=100,
        return_tensors="pt"
    ).to(device)

    h = enc(**enc_in).last_hidden_state
    amp_probs = torch.softmax(amp(h), dim=-1)[:, 1]
    tox_probs = torch.softmax(tox(h), dim=-1)[:, 1]

    # ===== save =====
    os.makedirs("results", exist_ok=True)
    with open("prodcal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "pAMP", "pTox"])
        for s, a, t in zip(seqs, amp_probs, tox_probs):
            w.writerow([s, f"{a.item():.6f}", f"{t.item():.6f}"])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, collater, tokenizer, scheme = OA_DM_38M()
    model = model.to(device).train()


    shared_encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    prot_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)

    toxic_model = ProteinHead(num_classes=2).to(device)
    amp_model = ProteinHead(num_classes=2).to(device)

    toxiccheckpoint = torch.load("best_toxic_model.pt", map_location=device)
    toxic_model.conv1.load_state_dict(toxiccheckpoint["conv1"])
    toxic_model.conv2.load_state_dict(toxiccheckpoint["conv2"])
    toxic_model.conv3.load_state_dict(toxiccheckpoint["conv3"])
    toxic_model.transformer.load_state_dict(toxiccheckpoint["transformer"])
    toxic_model.classifier.load_state_dict(toxiccheckpoint["classifier"])
    toxic_model.to(device)
    toxic_model.eval()

    ampcheckpoint = torch.load("head_only.pt", map_location=device)
    amp_model.conv1.load_state_dict(ampcheckpoint["conv1"])
    amp_model.conv2.load_state_dict(ampcheckpoint["conv2"])
    amp_model.conv3.load_state_dict(ampcheckpoint["conv3"])
    amp_model.transformer.load_state_dict(ampcheckpoint["transformer"])
    amp_model.classifier.load_state_dict(ampcheckpoint["classifier"])
    amp_model.to(device)
    amp_model.eval()

    fine_tune_config = {
        'batch_size': 1,
        'lr': 1e-5,  
        'weight_decay': 1e-4,
        'epochs': 1,
        'save_freq': 1
    }

    train_dataset = MinimalFastaDataset("positive_samples.fasta")
    fine_tune_oadm(model, train_dataset, tokenizer, collater, fine_tune_config)
    #known_amp_seqs = load_fasta("naturalAMPs_APD2024a.fasta")
    #warmup_supervised(model, tokenizer, known_amp_seqs, epochs=3)
    train_rl(num_steps=40, batch_size=400, max_len=30)
    generate_and_save(model, tokenizer,shared_encoder, prot_tokenizer, amp_model, toxic_model)
