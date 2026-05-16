import re
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


def _normalize(s: str) -> str:
    s = s.lower().strip().replace("ё", "е")
    s = re.sub(r"[^\w\sа-яА-Я\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _tokenize(s: str) -> List[str]:
    return [t for t in _normalize(s).split() if t]

def _token_overlap(source: str, target: str) -> float:
    s = set(_tokenize(source))
    t = set(_tokenize(target))
    if not s:
        return 0.0
    return len(s & t) / len(s)

def _token_f1(pred: str, gold: str) -> float:
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    counts = {}
    for t in p:
        counts[t] = counts.get(t, 0) + 1
    common = 0
    for t in g:
        if counts.get(t, 0) > 0:
            counts[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)

def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 5]


@dataclass
class RAGASScores:
    faithfulness: float
    context_relevancy: float
    context_recall: float
    answer_relevancy: float
    answer_correctness: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class LocalRAGAS:
    def __init__(self, index=None):
        self.index = index

    def _sem_sim(self, a: str, b: str) -> float:
        if self.index is None:
            return 0.0
        try:
            X = self.index.embed_texts(["passage: " + a, "passage: " + b], normalize=True)
            return float(X[0] @ X[1])
        except Exception:
            return 0.0

    def _ctx_sem_sim(self, texts: List[str], query: str) -> float:
        if self.index is None:
            return 0.0
        try:
            X = self.index.embed_texts(["passage: " + t for t in texts], normalize=True)
            q = self.index.embed_texts(["query: " + query], normalize=True)[0]
            return float(np.mean(X @ q))
        except Exception:
            return 0.0

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        sentences = _sentences(answer)
        if not sentences:
            return 1.0
        ctx = " ".join(contexts)
        supported = 0
        for sent in sentences:
            if _normalize(sent) in _normalize(ctx):
                supported += 1
                continue
            overlap = max(_token_overlap(sent, c) for c in contexts)
            sem = max(self._sem_sim(sent, c) for c in contexts)
            if overlap > 0.5 or sem > 0.7:
                supported += 1
        return supported / len(sentences)

    def context_relevancy(self, contexts: List[str], question: str) -> float:
        if not contexts:
            return 0.0
        sem = self._ctx_sem_sim(contexts, question)
        lex = np.mean([_token_overlap(question, c) for c in contexts])
        if sem > 0:
            return 0.6 * sem + 0.4 * float(lex)
        return float(lex)

    def context_recall(self, contexts: List[str], ground_truth: str) -> float:
        ctx = " ".join(contexts)
        sentences = _sentences(ground_truth)
        if not sentences:
            return _token_overlap(ground_truth, ctx)
        supported = 0
        for sent in sentences:
            overlap = _token_overlap(sent, ctx)
            sem = self._sem_sim(sent, ctx[:1000])
            if overlap > 0.4 or sem > 0.65:
                supported += 1
        return supported / len(sentences)

    def answer_relevancy(self, answer: str, question: str) -> float:
        sem = self._sem_sim(answer, question)
        lex = _token_overlap(question, answer)
        if sem > 0:
            return 0.7 * sem + 0.3 * lex
        return lex

    def answer_correctness(self, answer: str, ground_truth: str) -> float:
        f1 = _token_f1(answer, ground_truth)
        sem = self._sem_sim(answer, ground_truth)
        if sem > 0:
            return 0.5 * f1 + 0.5 * sem
        return f1

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> RAGASScores:
        return RAGASScores(
            faithfulness=self.faithfulness(answer, contexts),
            context_relevancy=self.context_relevancy(contexts, question),
            context_recall=self.context_recall(contexts, ground_truth),
            answer_relevancy=self.answer_relevancy(answer, question),
            answer_correctness=self.answer_correctness(answer, ground_truth),
        )

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Tuple[List[RAGASScores], Dict[str, float]]:
        results = [
            self.evaluate(q, a, c, g)
            for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        ]
        keys = list(results[0].as_dict().keys())
        agg = {k: float(np.mean([getattr(r, k) for r in results])) for k in keys}
        return results, agg