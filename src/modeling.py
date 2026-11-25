"""TweetLens: reproducible, CPU-only social text modeling."""
import re
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator

LABELS = ['negative', 'neutral', 'positive']

def tokens(text):
    return list(re.finditer(r"[A-Za-z]+(?:['’`][A-Za-z]+)?|[!?]+", str(text)))

def jaccard(a, b):
    x, y = set(str(a).lower().split()), set(str(b).lower().split())
    return len(x & y) / len(x | y) if x | y else 1.0

class EvidenceExtractor(BaseEstimator):
    """Supervised lexical span baseline; evidence is NOT a causal explanation.

    Learn smoothed per-sentiment log odds of a token appearing inside annotated
    spans versus elsewhere. Search contiguous token spans of at most 12 tokens.
    Neutral returns the original text, matching a common dataset convention.
    No validation annotations are used for fitting or parameter selection.
    """
    def __init__(self, max_tokens=12, penalty=0.35):
        self.max_tokens = max_tokens
        self.penalty = penalty

    def fit(self, frame):
        self.weights_ = {}
        self.skipped_ = 0
        for label in ['negative', 'positive']:
            inside, outside = Counter(), Counter()
            for row in frame[frame.sentiment == label].itertuples():
                start = row.text.find(row.selected_text)
                if start < 0:
                    self.skipped_ += 1
                    continue
                end = start + len(row.selected_text)
                for m in tokens(row.text):
                    (inside if m.start() < end and m.end() > start else outside)[m.group().lower()] += 1
            vocabulary = set(inside) | set(outside)
            ni, no, v = sum(inside.values()), sum(outside.values()), len(vocabulary)
            self.weights_[label] = {w: float(np.clip(np.log((inside[w]+1)/(ni+v)) - np.log((outside[w]+1)/(no+v)), -3, 3)) for w in vocabulary if inside[w]+outside[w] >= 3}
        return self

    def predict_one(self, text, label):
        if label == 'neutral' or len(str(text).split()) <= 2:
            return text
        ts = tokens(text)
        if not ts:
            return text
        weights = self.weights_.get(label, {})
        scores = [weights.get(t.group().lower(), 0) - self.penalty for t in ts]
        best = (-float('inf'), 0, 0)
        for i in range(len(ts)):
            total = 0
            for j in range(i, min(len(ts), i+self.max_tokens)):
                total += scores[j]
                if total > best[0]:
                    best = (total, i, j)
        if best[0] <= 0:
            return text
        return text[ts[best[1]].start():ts[best[2]].end()]
