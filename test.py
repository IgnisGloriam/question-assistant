import csv
import re
import math
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from embedder import LectureIndex
from chunk import chunk_text
from parser import extract_text
from retrieval import ContextRetriever
from retrieval.mmr_retrieval import MMRContextRetriever
from retrieval.dpp_retrieval import DPPContextRetriever
from generator import LLMGenerator
from ragas_metrics import LocalRAGAS



DOC_PATH      = "test_text/история.pdf"
TOPICS        = ["Когда родился Пётр Первый", "Екатерина Великая", "Российский флот"]
LLM_PATH      = "D:/models/qwen2.5-3b-instruct-q4_k_m.gguf"
METHODS       = ["topk"] #, "mmr", "dpp"
K_SELECT      = 5
TOP_M         = 15
NUM_QUESTIONS = 1
OUTPUT_CSV    = "question_numerical_metrics.csv"

_PUNCT = re.compile(r"[^\w\sа-яА-ЯёЁ\-]+", re.UNICODE)
_WS    = re.compile(r"\s+", re.UNICODE)

def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()

def _tok(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t]


def _lcs(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

def rouge_l(pred: str, gold: str, beta: float = 1.2) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    lcs = _lcs(p, g)
    if lcs == 0: return 0.0
    pr, rc = lcs / len(p), lcs / len(g)
    if pr + rc == 0: return 0.0
    return float(((1 + beta**2) * pr * rc) / (rc + beta**2 * pr))

def _ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    c = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        c[ng] = c.get(ng, 0) + 1
    return c

def bleu_n(pred: str, gold: str, max_n: int = 4, smooth: float = 1e-9) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    precs = []
    for n in range(1, max_n + 1):
        pc, gc = _ngrams(p, n), _ngrams(g, n)
        ov = sum(min(cnt, gc.get(ng, 0)) for ng, cnt in pc.items())
        tot = sum(pc.values())
        precs.append((ov + smooth) / (tot + smooth))
    log_p = sum(math.log(x) for x in precs) / max_n
    bp = math.exp(1.0 - len(g) / max(1, len(p))) if len(p) < len(g) else 1.0
    return float(bp * math.exp(log_p))

def _char_ngrams(s: str, n: int) -> Dict[str, int]:
    s = _norm(s).replace(" ", "")
    c = {}
    for i in range(len(s) - n + 1):
        ng = s[i:i+n]
        c[ng] = c.get(ng, 0) + 1
    return c

def chrf(pred: str, gold: str, n: int = 6, beta: float = 2.0) -> float:
    if not pred and not gold: return 1.0
    if not pred or not gold:  return 0.0
    pn, gn = _char_ngrams(pred, n), _char_ngrams(gold, n)
    ov = sum(min(cnt, gn.get(ng, 0)) for ng, cnt in pn.items())
    pt, gt = sum(pn.values()), sum(gn.values())
    if pt == 0 or gt == 0: return 0.0
    pr, rc = ov / pt, ov / gt
    if pr + rc == 0: return 0.0
    return float((1 + beta**2) * pr * rc / (beta**2 * pr + rc))

def jaccard(pred: str, gold: str) -> float:
    ps, gs = set(_tok(pred)), set(_tok(gold))
    if not ps and not gs: return 1.0
    if not ps or not gs:  return 0.0
    return float(len(ps & gs) / len(ps | gs))

def levenshtein_sim(pred: str, gold: str) -> float:
    a, b = _norm(pred), _norm(gold)
    if not a and not b: return 1.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
            prev = cur
    return float(1.0 - dp[m] / max(1, max(n, m)))

def context_support(pred: str, context: str) -> float:
    np_ = _norm(pred)
    nc  = _norm(context)
    if not np_: return 0.0
    return 1.0 if np_ in nc else 0.0

def max_sent_overlap(pred: str, context: str) -> float:
    ans = set(_tok(pred))
    if not ans or not context: return 0.0
    best = 0.0
    for s in re.split(r"(?<=[.!?])\s+", context.strip()):
        st = set(_tok(s))
        if st:
            best = max(best, len(ans & st) / len(ans | st))
    return float(best)




def self_bleu(questions: List[str], max_n: int = 4) -> float:
    """
    Self-BLEU: для каждого вопроса считаем BLEU, где остальные — references.
    Чем ниже self-BLEU, тем разнообразнее вопросы.
    """
    if len(questions) < 2:
        return 0.0
    smooth = SmoothingFunction().method1
    scores = []
    for i, hyp in enumerate(questions):
        refs = [_tok(q) for j, q in enumerate(questions) if j != i]
        hyp_tok = _tok(hyp)
        if hyp_tok:
            scores.append(
                sentence_bleu(refs, hyp_tok, smoothing_function=smooth)
            )
    return float(np.mean(scores)) if scores else 0.0



def _q_emb(index: LectureIndex, text: str) -> np.ndarray:
    return index.embed_texts(["query: " + text], normalize=True)[0]

def _p_emb(index: LectureIndex, text: str) -> np.ndarray:
    return index.embed_texts(["passage: " + text], normalize=True)[0]

def compute_relevance(question: str, context: str, index: LectureIndex) -> float:
    q = _q_emb(index, question).reshape(1, -1)
    c = _p_emb(index, context).reshape(1, -1)
    return float(cosine_similarity(q, c)[0][0])

def compute_faithfulness(question: str, chunks: List[str], index: LectureIndex) -> float:
    q    = _q_emb(index, question).reshape(1, -1)
    embs = index.embed_texts(["passage: " + c for c in chunks], normalize=True)
    return float(np.max(cosine_similarity(q, embs)))

def compute_novelty(question: str, others: List[str], index: LectureIndex) -> float:
    rest = [o for o in others if o != question]
    if not rest: return 1.0
    q    = _q_emb(index, question).reshape(1, -1)
    embs = np.vstack([_q_emb(index, o) for o in rest])
    return float(1.0 - np.mean(cosine_similarity(q, embs)))



def compute_coverage(question: str, context: str) -> float:
    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
    c_words = set(re.findall(r"\b\w{4,}\b", context.lower()))
    if not q_words: return 0.0
    return len(q_words & c_words) / len(q_words)

def compute_complexity(question: str) -> float:
    words = re.findall(r"\b\w+\b", question.lower())
    if len(words) < 2: return 0.0
    return min(1.0, (np.mean([len(w) for w in words]) + np.std([len(w) for w in words])) / 15.0)

def compute_diversity(questions: List[str]) -> float:
    """1 - mean pairwise BLEU (NLTK). Чем выше — тем разнообразнее."""
    if len(questions) < 2: return 1.0
    smooth = SmoothingFunction().method1
    scores = []
    for i, q1 in enumerate(questions):
        for j, q2 in enumerate(questions):
            if i != j:
                scores.append(
                    sentence_bleu([q2.split()], q1.split(), smoothing_function=smooth)
                )
    return 1.0 - (sum(scores) / len(scores))



def compute_question_answer_metrics(question: str, context: str) -> Dict[str, float]:
    """
    Считаем метрики из answer_metrics.py, трактуя вопрос как pred, контекст как gold.
    Это позволяет измерить, насколько вопрос "вытекает" из контекста.
    """
    return {
        "rouge_l":             rouge_l(question, context),
        "bleu4":               bleu_n(question, context, max_n=4),
        "chrf":                chrf(question, context),
        "jaccard":             jaccard(question, context),
        "levenshtein_sim":     levenshtein_sim(question, context),
        "context_support":     context_support(question, context),
        "max_sent_overlap":    max_sent_overlap(question, context),
    }


# -------------------------
# RAGAS-based metrics
# -------------------------

def compute_ragas(
    questions: List[str],
    chunks: List[str],
    topic: str,
    index: LectureIndex,
) -> Dict[str, float]:
    ragas = LocalRAGAS(index=index)
    return {
        "ragas_faithfulness":     np.mean([ragas.faithfulness(q, chunks)         for q in questions]),
        "ragas_ctx_relevancy":    np.mean([ragas.context_relevancy(chunks, q)    for q in questions]),
        "ragas_answer_relevancy": np.mean([ragas.answer_relevancy(q, topic)      for q in questions]),
    }


# -------------------------
# Retrieval
# -------------------------

class RetrievalMethod:
    def __init__(self, name: str, index: LectureIndex):
        self.name = name
        self.index = index
        if name == "topk":
            self.retriever = ContextRetriever(index)
        elif name == "mmr":
            self.retriever = MMRContextRetriever(index)
        elif name == "dpp":
            self.retriever = DPPContextRetriever(index)
        else:
            raise ValueError(f"Unknown method: {name}")

    def retrieve(
        self, query: str, n_results: int = 5, top_m: int = 15
    ) -> Tuple[str, List[str], List[str]]:
        if self.name == "topk":
            context    = self.retriever.get_context(topic=query, n_results=n_results, max_context_chars=3000)
            selected   = self.index.search(query, n_results=n_results)
            candidates = self.index.search(query, n_results=top_m)
        elif self.name == "mmr":
            context, selected = self.retriever.get_context(
                query=query, top_m=top_m, n_select=n_results, lambda_mult=0.6, max_context_chars=3000
            )
            candidates = self.index.search(query, n_results=top_m)
        else:
            context, selected = self.retriever.get_context(
                topic=query, top_m=top_m, n_select=n_results, alpha=5.0, max_context_chars=3000
            )
            candidates = self.index.search(query, n_results=top_m)
        return context, selected, candidates


# -------------------------
# Question generation
# -------------------------

class QuestionGenerator:
    def __init__(self, llm: LLMGenerator):
        self.llm = llm

    def generate(self, context: str, topic: str, num_questions: int = 3) -> List[str]:
        temperatures = [0.5, 0.6, 0.7, 0.8, 0.9]
        questions    = []

        for i in range(num_questions):
            temperature = temperatures[i % len(temperatures)]
            
            prompt = (
                f"""<|im_start|>system
Ты — опытный методист и помощник преподавателя. Твоя задача — составлять качественные тестовые вопросы строго на основе предоставленного учебного материала.
Не придумывай факты, которых нет в тексте.<|im_end|>
<|im_start|>user
Составь тестовый вопрос на тему {topic} с 4 вариантами ответа на основе материала ниже.

МАТЕРИАЛ ДЛЯ ВОПРОСОВ:
{context}

ТРЕБОВАНИЯ И ОГРАНИЧЕНИЯ:
1. Вопрос должен проверять одну конкретную и важную идею из материала.
2. Правильный ответ должен быть только один и абсолютно однозначный.
3. Неправильные варианты (дистракторы) должны быть правдоподобными и логичными, чтобы студент, не знающий материала, мог ошибиться.
4. Не используй формулировки типа «Все вышеперечисленное» или «Ничего из вышеперечисленного».
5. Строго опирайся только на предоставленный материал. Если для ответа нужны знания извне — не задавай этот вопрос.

ФОРМАТ ОТВЕТА (строго соблюдай):
**Вопрос:** [Текст вопроса]
А) [Вариант 1]
Б) [Вариант 2]
В) [Вариант 3]
Г) [Вариант 4]
✅ Правильный ответ: [Буква]
💡 Пояснение: [Краткое объяснение из текста, почему ответ верен]


Ответ должен быть на русском языке!

Начинай генерировать вопрос:<|im_end|>
<|im_start|>assistant
"""
            )
            print(prompt)
            raw = self.llm.generate_single(prompt, temperature=temperature).strip()
            if raw:
                questions.append(raw)

        return questions


# -------------------------
# CSV fieldnames
# -------------------------

FIELDNAMES = [
    "topic", "retrieval_method", "num_questions",
    
    "relevance_mean",    "relevance_std",
    "faithfulness_mean", "faithfulness_std",
    "novelty_mean",      "novelty_std",
    
    "coverage_mean",     "coverage_std",
    "complexity_mean",   "complexity_std",
    "diversity",
    
    "self_bleu",
    
    "rouge_l_mean",          "bleu4_mean",
    "chrf_mean",             "jaccard_mean",
    "levenshtein_sim_mean",  "context_support_mean",
    "max_sent_overlap_mean",
    
    "ragas_faithfulness", "ragas_ctx_relevancy", "ragas_answer_relevancy",

    "generated_questions",
]


# -------------------------
# Main pipeline
# -------------------------

def evaluate_question_pipeline() -> None:
    text   = extract_text(DOC_PATH)
    chunks = chunk_text(text)
    index  = LectureIndex()
    index.index_chunks(chunks)

    llm       = LLMGenerator(LLM_PATH)
    generator = QuestionGenerator(llm)
    results   = []

    for t_idx, topic in enumerate(TOPICS):
        print(f"\nTopic {t_idx+1}/{len(TOPICS)}: {topic}")

        for method_name in METHODS:
            print(f"  [{method_name}]...", end=" ", flush=True)

            retriever                     = RetrievalMethod(method_name, index)
            context, selected, candidates = retriever.retrieve(
                topic, n_results=K_SELECT, top_m=TOP_M
            )
            questions = generator.generate(context, topic, num_questions=NUM_QUESTIONS)
            print(questions)

            if not questions:
                print("no questions generated")
                continue

            # embedding-based
            rel = [compute_relevance(q, context, index)    for q in questions]
            fth = [compute_faithfulness(q, selected, index) for q in questions]
            nov = [compute_novelty(q, questions, index)     for q in questions]

            # text-based
            cov = [compute_coverage(q, context)   for q in questions]
            cpx = [compute_complexity(q)           for q in questions]
            div = compute_diversity(questions)
            sb  = self_bleu(questions)

            # per-question answer metrics
            am_keys = ["rouge_l", "bleu4", "chrf", "jaccard",
                       "levenshtein_sim", "context_support", "max_sent_overlap"]
            am_lists: Dict[str, List[float]] = {k: [] for k in am_keys}
            for q in questions:
                m = compute_question_answer_metrics(q, context)
                for k in am_keys:
                    am_lists[k].append(m[k])

            # ragas
            rgs = compute_ragas(questions, selected, topic, index)

            row = {
                "topic":                  topic,
                "retrieval_method":       method_name,
                "num_questions":          len(questions),
                "relevance_mean":         np.mean(rel),
                "relevance_std":          np.std(rel),
                "faithfulness_mean":      np.mean(fth),
                "faithfulness_std":       np.std(fth),
                "novelty_mean":           np.mean(nov),
                "novelty_std":            np.std(nov),
                "coverage_mean":          np.mean(cov),
                "coverage_std":           np.std(cov),
                "complexity_mean":        np.mean(cpx),
                "complexity_std":         np.std(cpx),
                "diversity":              div,
                "self_bleu":              sb,
                "rouge_l_mean":           np.mean(am_lists["rouge_l"]),
                "bleu4_mean":             np.mean(am_lists["bleu4"]),
                "chrf_mean":              np.mean(am_lists["chrf"]),
                "jaccard_mean":           np.mean(am_lists["jaccard"]),
                "levenshtein_sim_mean":   np.mean(am_lists["levenshtein_sim"]),
                "context_support_mean":   np.mean(am_lists["context_support"]),
                "max_sent_overlap_mean":  np.mean(am_lists["max_sent_overlap"]),
                **rgs,
                "generated_questions":    " || ".join(q.splitlines()[0] for q in questions[:2]),
            }
            results.append(row)
            print(f"done ({len(questions)} questions)")

    if not results:
        print("No results generated.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    print(f"\nSaved to {OUTPUT_CSV}")

    agg_metrics = [
        "relevance_mean", "faithfulness_mean", "novelty_mean",
        "coverage_mean", "complexity_mean", "diversity", "self_bleu",
        "rouge_l_mean", "bleu4_mean", "chrf_mean", "jaccard_mean",
        "levenshtein_sim_mean", "context_support_mean", "max_sent_overlap_mean",
        "ragas_faithfulness", "ragas_ctx_relevancy", "ragas_answer_relevancy",
    ]

    print("\n" + "=" * 70)
    print("AGGREGATED METRICS")
    print("=" * 70)
    for method in METHODS:
        rows = [r for r in results if r["retrieval_method"] == method]
        if not rows:
            continue
        print(f"\n{method.upper()}:")
        for m in agg_metrics:
            vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
            if vals:
                print(f"  {m:35s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")


if __name__ == "__main__":
    evaluate_question_pipeline()