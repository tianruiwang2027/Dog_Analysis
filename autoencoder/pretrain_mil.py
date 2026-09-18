"""
Training loop for MILQueryMatcher (teacher/mil_matcher.py), and the matching
auto-labeling function that plays the role teacher/autolabel.py plays for the
original TeacherMatcher. Reuses teacher/pretrain_matcher.py's extract_features
unchanged -- same Stage-1 candidate generator, same Pan-Tompkins/annotation-hinted
beat times, same automatic clock alignment, same "no hand-clicked SCG label anywhere"
guarantee. The only new inputs are which DOMAIN each recording belongs to (so the
model can condition on it) and the reject-aware target assignment in mil_matcher.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .mil_matcher import DOMAIN_TO_ID, MILQueryMatcher, assign_targets
from .pretrain_matcher import RecordingFeatures, S1_SEARCH_HI_US, S1_SEARCH_LO_US, extract_features
from common.real_data import RealRecording

# dog_noninvasive is the only domain where ECG and SCG genuinely come from separate
# acquisition devices with an unknown clock offset (Chelten/Chuck/Dasty, per
# PIPELINE_SUMMARY.md); the epicardial JSON files and CEBS WFDB records store ECG and
# accelerometer already sample-synchronized in one file, so estimating a lag for them
# from a short/regular recording risks fitting noise rather than a real offset -- force
# it to exactly 0 for those domains instead of letting clock_align guess.
_ZERO_LAG_DOMAINS = {"dog_epicardial", "human"}


def _extract(rec: RealRecording) -> RecordingFeatures:
    lag_override = 0.0 if rec.domain in _ZERO_LAG_DOMAINS else None
    return extract_features(rec, lag_us_override=lag_override, adaptive_refract=True)


def mil_loss(model: MILQueryMatcher, feats: RecordingFeatures, domain_id: int, device: str = "cpu"):
    if len(feats.snippets) == 0 or len(feats.beat_times) == 0:
        return None, 0

    snippets_t = torch.tensor(feats.snippets, device=device).unsqueeze(1)
    rr_t = torch.tensor(feats.rr_context, device=device)
    cand_emb = model.embed_candidates(snippets_t, feats.cand_t_ecg_clock, domain_id)

    delta = feats.cand_t_ecg_clock[None, :] - feats.beat_times[:, None]
    in_window = (delta >= S1_SEARCH_LO_US) & (delta <= S1_SEARCH_HI_US)
    valid, targets = assign_targets(delta, in_window)
    if not valid.any():
        return None, 0

    losses = []
    for i in np.where(valid)[0]:
        elig = np.where(in_window[i])[0]
        query = model.beat_query(rr_t[i:i + 1], domain_id).squeeze(0)
        logits = model.beat_logits(query, cand_emb[elig])
        target = torch.tensor(targets[i], device=device, dtype=torch.long)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))

    total = torch.stack(losses).mean()
    return total, len(losses)


def train_mil_matcher(
    recordings: list[RealRecording],
    n_epochs: int = 15,
    lr: float = 3e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[MILQueryMatcher, list[RecordingFeatures]]:
    model = MILQueryMatcher().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    feats_list = [_extract(r) for r in recordings]
    domain_ids = [DOMAIN_TO_ID[r.domain] for r in recordings]
    if verbose:
        for f, r in zip(feats_list, recordings):
            print(f"  [{r.domain:16s}] {r.dog:20s} lag={f.estimated_lag_us/1e6:+.2f}s "
                  f"{len(f.snippets)} candidates, {len(f.beat_times)} ECG beats")

    for epoch in range(n_epochs):
        epoch_losses = []
        for feats, did in zip(feats_list, domain_ids):
            loss, n_beats = mil_loss(model, feats, did, device)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            print(f"  epoch {epoch+1}/{n_epochs}  mean loss={np.mean(epoch_losses):.4f}")

    return model, feats_list


@dataclass
class MilAutoLabelResult:
    dog: str
    snippets: np.ndarray
    s1_labels: np.ndarray
    n_beats_matched: int
    n_beats_total: int
    n_beats_rejected: int
    estimated_lag_us: float


@torch.no_grad()
def auto_label_recording_mil(
    model: MILQueryMatcher, rec: RealRecording, domain_id: int, device: str = "cpu"
) -> tuple[MilAutoLabelResult, RecordingFeatures]:
    feats = _extract(rec)
    n_cand = len(feats.snippets)
    n_beats = len(feats.beat_times)
    s1_labels = np.zeros(n_cand, dtype=np.float32)
    n_matched = 0
    n_rejected = 0

    if n_cand and n_beats:
        snippets_t = torch.tensor(feats.snippets, device=device).unsqueeze(1)
        rr_t = torch.tensor(feats.rr_context, device=device)
        cand_emb = model.embed_candidates(snippets_t, feats.cand_t_ecg_clock, domain_id)

        delta = feats.cand_t_ecg_clock[None, :] - feats.beat_times[:, None]
        in_window = (delta >= S1_SEARCH_LO_US) & (delta <= S1_SEARCH_HI_US)

        for i in range(n_beats):
            elig = np.where(in_window[i])[0]
            if len(elig) == 0:
                continue
            query = model.beat_query(rr_t[i:i + 1], domain_id).squeeze(0)
            logits = model.beat_logits(query, cand_emb[elig])
            pick = int(torch.argmax(logits).item())
            if pick == len(elig):  # null -- model abstained
                n_rejected += 1
                continue
            s1_labels[elig[pick]] = 1.0
            n_matched += 1

    result = MilAutoLabelResult(
        dog=rec.dog, snippets=feats.snippets, s1_labels=s1_labels,
        n_beats_matched=n_matched, n_beats_total=n_beats, n_beats_rejected=n_rejected,
        estimated_lag_us=feats.estimated_lag_us,
    )
    return result, feats
