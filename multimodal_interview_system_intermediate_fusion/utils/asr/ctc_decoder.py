"""
CTC decoder — greedy and beam search.

Both work on log-probability tensors output by the CTC head.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple


# =========================================================================== #
#  Greedy (fastest, ~1% worse WER than beam search)                            #
# =========================================================================== #

def greedy_decode(
    log_probs: torch.Tensor,
    lengths:   torch.Tensor,
    blank_id:  int = 0,
) -> List[List[int]]:
    """
    Greedy CTC decode — argmax at each time step, collapse repeats, drop blanks.

    Args:
        log_probs: [B, T, vocab_size]  (log-softmax output)
        lengths:   [B] actual (subsampled) time lengths per sample
        blank_id:  index of the CTC blank token

    Returns:
        List of decoded token-id sequences (one per batch item)
    """
    argmax = log_probs.argmax(dim=-1)   # [B, T]
    results: List[List[int]] = []

    for b in range(argmax.size(0)):
        seq_len = lengths[b].item() if lengths is not None else argmax.size(1)
        tokens  = argmax[b, :seq_len].tolist()

        # Collapse repeats then remove blanks
        collapsed: List[int] = []
        prev = None
        for t in tokens:
            if t != prev:
                if t != blank_id:
                    collapsed.append(t)
                prev = t

        results.append(collapsed)

    return results


# =========================================================================== #
#  Beam Search                                                                 #
# =========================================================================== #

class BeamEntry:
    """A single beam hypothesis."""
    __slots__ = ("tokens", "score_blank", "score_no_blank")

    def __init__(self, tokens: tuple = (), score_blank: float = 0.0, score_no_blank: float = float("-inf")):
        self.tokens        = tokens
        self.score_blank   = score_blank
        self.score_no_blank = score_no_blank

    @property
    def total(self) -> float:
        import math
        if self.score_blank == float("-inf") and self.score_no_blank == float("-inf"):
            return float("-inf")
        if self.score_blank == float("-inf"):
            return self.score_no_blank
        if self.score_no_blank == float("-inf"):
            return self.score_blank
        m = max(self.score_blank, self.score_no_blank)
        return m + math.log(
            math.exp(self.score_blank - m) + math.exp(self.score_no_blank - m)
        )


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    import math
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search_decode(
    log_probs: torch.Tensor,
    lengths:   torch.Tensor,
    blank_id:  int = 0,
    beam_width: int = 10,
) -> List[List[int]]:
    """
    Prefix beam search CTC decode (pure Python, single-sample loop).

    Args:
        log_probs:  [B, T, vocab_size]
        lengths:    [B] actual time lengths
        blank_id:   CTC blank token index
        beam_width: number of beams to keep

    Returns:
        List of best token-id sequences (one per batch item)
    """
    import math

    batch_results: List[List[int]] = []
    B, T_max, V = log_probs.shape
    lp = log_probs.cpu().tolist()   # move to CPU for loop efficiency

    for b in range(B):
        seq_len = int(lengths[b].item()) if lengths is not None else T_max

        # Initial beam: empty prefix, blank score = 0, no-blank = -inf
        beams: dict = {(): BeamEntry((), score_blank=0.0)}

        for t in range(seq_len):
            new_beams: dict = {}

            for prefix, entry in beams.items():
                last = prefix[-1] if prefix else None

                for v in range(V):
                    lp_tv = lp[b][t][v]

                    if v == blank_id:
                        # Emit blank → extend same prefix
                        key = prefix
                        nb  = new_beams.setdefault(key, BeamEntry(key))
                        nb.score_blank = _log_add(
                            nb.score_blank,
                            entry.total + lp_tv
                        )
                    else:
                        new_prefix = prefix + (v,)

                        if v == last:
                            # Same char as last in prefix: only via blank
                            key = new_prefix
                            nb  = new_beams.setdefault(key, BeamEntry(key))
                            nb.score_no_blank = _log_add(
                                nb.score_no_blank,
                                entry.score_blank + lp_tv
                            )
                        else:
                            key = new_prefix
                            nb  = new_beams.setdefault(key, BeamEntry(key))
                            nb.score_no_blank = _log_add(
                                nb.score_no_blank,
                                entry.total + lp_tv
                            )

            # Prune to top beam_width
            beams = dict(
                sorted(new_beams.items(), key=lambda kv: kv[1].total, reverse=True)[:beam_width]
            )

        # Best beam
        best = max(beams.values(), key=lambda e: e.total)
        batch_results.append(list(best.tokens))

    return batch_results
