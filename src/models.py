"""BDHCore and BDHModel."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.config import ENCODER_NAME, MAX_TOKENS, NUM_BELIEF_PATHS, STATE_DIM
from src.utils import sentence_split


class BDHCore(nn.Module):
    """Gated belief-state update. Returns (new_state, friction), where
    friction is the L2 norm of the applied update."""

    def __init__(self, dim):
        super().__init__()
        self.update = nn.Linear(dim * 2, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, state, event):
        combined = torch.cat([state, event], dim=-1)
        delta = torch.tanh(self.update(combined))
        gate = self.gate(combined)
        update = gate * delta
        friction = torch.norm(update, p=2, dim=-1, keepdim=True)
        return state + update, friction


class BDHModel(nn.Module):
    """Frozen sentence encoder + BDH core + classification head."""

    def __init__(self, encoder_name=ENCODER_NAME):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.bdh = BDHCore(STATE_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(STATE_DIM + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    @property
    def device(self):
        """Current device of the model (follows .to(device) calls)."""
        return next(self.parameters()).device

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.encoder(**inputs)
        return out.last_hidden_state.mean(dim=1)

    def forward(self, captions, contents, return_friction_stats=False):
        B = len(captions)
        device = self.device

        base_state = self.embed(list(captions))
        belief_states = base_state.unsqueeze(1).repeat(1, NUM_BELIEF_PATHS, 1)

        avg_frictions = torch.zeros(B, device=device)
        var_frictions = torch.zeros(B, device=device)

        for i in range(B):
            sentences = sentence_split(contents[i])
            sample_frictions = []
            current_state = belief_states[i]

            for sent in sentences:
                event = self.embed(sent).repeat(NUM_BELIEF_PATHS, 1)
                current_state, friction = self.bdh(current_state, event)
                sample_frictions.append(friction.squeeze(1))

            belief_states[i] = current_state

            if sample_frictions:
                all_f = torch.cat(sample_frictions, dim=0)
                avg_frictions[i] = all_f.mean()
                var_frictions[i] = all_f.var(unbiased=False)

        pooled_state = belief_states.mean(dim=1)
        meta = torch.stack([avg_frictions, var_frictions], dim=1)
        logits = self.classifier(torch.cat([pooled_state, meta], dim=-1))

        friction_sum = avg_frictions + var_frictions
        aux_loss = friction_sum.mean()

        if return_friction_stats:
            return logits, aux_loss, avg_frictions, var_frictions, friction_sum
        return logits, aux_loss, friction_sum
