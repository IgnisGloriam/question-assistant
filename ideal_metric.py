import os
import csv
import re
import math
import time
import random
import gc
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

from embedder import LectureIndex
from chunk import chunk_text
from retrieval import ContextRetriever
from generator import LLMGenerator
from ragas_metrics import LocalRAGAS



N_XQUAD_SAMPLES = 100
K_SELECT = 3        

LLM_PATHS = [
    #"D:/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    "D:/models/qwen3.5-2b-q4_k_m.gguf",
    #"D:/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    #"D:/models/Qwen3-1.7B-Q4_K_M-Instruct.gguf",
]

OUTPUT_CSV = "rag_models_comparison.csv"
OUTPUT_MD = "rag_models_comparison_log.md"

FIELDNAMES = [
    "model_name", "example_id", "xquad_original_question", "generated_question",
    "rouge_l", "bleu4", "semantic_sim", 
    "distractor_distinctness", "distractor_plausibility", "format_adherence",
    "ragas_faithfulness", "ragas_answer_relevancy",
    "generation_time_sec", "xquad_gold_answer", "llm_correct_answer"
]


def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = re.sub(r"[^\w\sа-яА-ЯёЁ\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _tok(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t]

def rouge_l(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    n, m = len(p), len(g)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if p[i-1] == g[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    lcs = dp[m]
    if lcs == 0: return 0.0
    pr, rc = lcs / len(p), lcs / len(g)
    return float(((1 + 1.2**2) * pr * rc) / (rc + 1.2**2 * pr)) if pr + rc > 0 else 0.0

def bleu_4(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p or not g:  return 0.0
    precs = []
    for n in range(1, 5):
        pc = {tuple(p[i:i+n]): 1 for i in range(len(p)-n+1)}
        gc = {tuple(g[i:i+n]): 1 for i in range(len(g)-n+1)}
        ov = sum(1 for ng in pc if ng in gc)
        precs.append((ov + 1e-9) / (len(pc) + 1e-9))
    log_p = sum(math.log(x) for x in precs) / 4
    bp = math.exp(1.0 - len(g) / max(1, len(p))) if len(p) < len(g) else 1.0
    return float(bp * math.exp(log_p))

def semantic_similarity(pred: str, gold: str, index: LectureIndex) -> float:
    if not pred.strip() or not gold.strip(): return 0.0
    p_emb = index.embed_texts([pred], normalize=True)[0].reshape(1, -1)
    g_emb = index.embed_texts([gold], normalize=True)[0].reshape(1, -1)
    return float(cosine_similarity(p_emb, g_emb)[0][0])

def distractor_distinctness(correct_ans: str, distractors: List[str], index: LectureIndex) -> float:
    if not correct_ans or not distractors: return 0.0
    c_emb = index.embed_texts([correct_ans], normalize=True)[0].reshape(1, -1)
    d_embs = index.embed_texts(distractors, normalize=True)
    sims = cosine_similarity(c_emb, d_embs)[0]
    return float(1.0 - np.mean(sims))

def distractor_plausibility(question: str, distractors: List[str], index: LectureIndex) -> float:
    if not question or not distractors: return 0.0
    q_emb = index.embed_texts([question], normalize=True)[0].reshape(1, -1)
    d_embs = index.embed_texts(distractors, normalize=True)
    sims = cosine_similarity(q_emb, d_embs)[0]
    return float(np.mean(sims))

def format_adherence_score(question: str, options: List[str], correct_ans: str) -> float:
    score = 0.0
    if question.strip(): score += 0.25
    if len(options) == 4: score += 0.25
    if correct_ans.strip(): score += 0.25
    if correct_ans and any(_norm(correct_ans) in _norm(opt) or _norm(opt) in _norm(correct_ans) for opt in options):
        score += 0.25
    return float(score)

def compute_ragas(question: str, pred_answer: str, chunks: List[str], index: LectureIndex) -> Dict[str, float]:
    ragas = LocalRAGAS(index=index)
    return {
        "ragas_faithfulness":     float(ragas.faithfulness(pred_answer, chunks)),
        "ragas_answer_relevancy": float(ragas.answer_relevancy(question, pred_answer)),
    }


def build_test_prompt(context: str, topic: str) -> str:
    return f"""<|im_start|>system
Ты — строгий экзаменатор. Составь ОДИН тестовый вопрос по предоставленному тексту.
Тема вопроса: {topic}

ПРАВИЛА:
1. Придумай ровно 4 РАЗНЫХ варианта ответа. Варианты НЕ ДОЛЖНЫ повторяться!
2. В строке "Правильный ответ:" напиши САМ ТЕКСТ ответа, а не просто букву.

Формат ответа СТРОГО следующий (без лишних рассуждений):
Вопрос: [Текст вопроса]
А) [Вариант 1]
Б) [Вариант 2]
В) [Вариант 3]
Г) [Вариант 4]
Правильный ответ: [Текст правильного варианта]<|im_end|>
<|im_start|>user
КОНТЕКСТ:
{context}<|im_end|>
<|im_start|>assistant
"""


def parse_generated_test(raw_text: str) -> Tuple[str, List[str], List[str], str]:
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    
    lines = clean_text.split('\n')
    question = ""
    options = []
    correct_answer = ""

    for line in lines:
        line = line.strip()
        if not line: continue

        if line.lower().startswith("вопрос:"):
            question = line.split(":", 1)[1].strip()
        elif re.match(r"^[А-ГA-Dа-гa-d][\)\.]", line):
            opt_text = re.sub(r"^[А-ГA-Dа-гa-d][\)\.]\s*", "", line)
            options.append(opt_text)
        elif "правильный ответ:" in line.lower():
            ans = line.split(":", 1)[1].strip()
            ans = re.sub(r"^[А-ГA-Dа-гa-d][\)\.]\s*", "", ans)
            correct_answer = ans
    
    if not question and lines: question = lines[0]
    if not correct_answer and lines: correct_answer = lines[-1]

    distractors = [opt for opt in options if _norm(opt) not in _norm(correct_answer)]

    return question, options, distractors, correct_answer


if __name__ == "__main__":

    print("Загрузка и перемешивание XQuAD (ru)...")
    ds = load_dataset("xquad", "xquad.ru")["validation"]
    seed = 40 #random.randint(1, 10000)
    print(seed)
    ds_shuffled = ds.shuffle(seed=seed)
    
    n_limit = min(N_XQUAD_SAMPLES, len(ds_shuffled))
    selected_examples = [ds_shuffled[i] for i in range(n_limit)]
    
    results = []
    
    md_logs = {i: {"xquad_q": ex["question"], "gold": ex["answers"]["text"][0] if ex["answers"]["text"] else "", "models": {}} for i, ex in enumerate(selected_examples)}

    for model_path in LLM_PATHS:
        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            continue
            
        model_name = os.path.basename(model_path)
        print(f"\n{'='*60}")
        print(f"🚀 ИНИЦИАЛИЗАЦИЯ МОДЕЛИ: {model_name}")
        print(f"{'='*60}")
        
        generator = LLMGenerator(model_path=model_path)
        
        for i, ex in enumerate(selected_examples):
            context_full = ex["context"]
            xquad_q = ex["question"]
            gold_answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

            print(f"\n[{i+1}/{n_limit}] Обработка темой: {xquad_q}")
            
            chunks = chunk_text(context_full, min_size=200, max_size=500, overlap=50)
            current_index = LectureIndex()
            current_index.index_chunks(chunks)

            retriever = ContextRetriever(current_index)
            retrieved_context = retriever.get_context(topic=xquad_q, n_results=K_SELECT, max_context_chars=2000)
            retrieved_chunks = current_index.search(xquad_q, n_results=K_SELECT)

            t0 = time.time()
            prompt = build_test_prompt(retrieved_context, topic=xquad_q)

            if ("qwen3.5" in model_path):
                print("bp")
                prompt += "<think> </think>"

            raw_output = generator.generate_candidates(prompt, n_candidates=1,temperatures=[0.3])[0]
            elapsed = time.time() - t0

            gen_q, all_options, distractors, llm_correct_ans = parse_generated_test(raw_output)

            r_l = rouge_l(llm_correct_ans, gold_answer)
            b_4 = bleu_4(llm_correct_ans, gold_answer)
            s_sim = semantic_similarity(llm_correct_ans, gold_answer, current_index)
            
            d_dist = distractor_distinctness(llm_correct_ans, distractors, current_index)
            d_plaus = distractor_plausibility(gen_q, distractors, current_index) 
            f_score = format_adherence_score(gen_q, all_options, llm_correct_ans) 
            
            rgs = compute_ragas(gen_q, llm_correct_ans, retrieved_chunks, current_index)

            print(f"  Время: {elapsed:.1f}s | Format Score: {f_score:.2f} | Sem Sim: {s_sim:.2f}")

            results.append({
                "model_name": model_name,
                "example_id": ex["id"],
                "xquad_original_question": xquad_q,
                "generated_question": gen_q,
                "rouge_l": r_l,
                "bleu4": b_4,
                "semantic_sim": s_sim,
                "distractor_distinctness": d_dist,
                "distractor_plausibility": d_plaus,
                "format_adherence": f_score,
                "ragas_faithfulness": rgs["ragas_faithfulness"],
                "ragas_answer_relevancy": rgs["ragas_answer_relevancy"],
                "generation_time_sec": elapsed,
                "xquad_gold_answer": gold_answer,
                "llm_correct_answer": llm_correct_ans
            })

            md_logs[i]["models"][model_name] = {
                "raw_output": raw_output.strip(),
                "llm_correct_ans": llm_correct_ans,
                "s_sim": s_sim,
                "d_dist": d_dist,
                "d_plaus": d_plaus,
                "f_score": f_score,
                "faithfulness": rgs["ragas_faithfulness"]
            }

        del generator
        gc.collect()


    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\n✅ Результаты генерации тестов сохранены в {OUTPUT_CSV}")

        with open(OUTPUT_MD, "w", encoding="utf-8") as md_file:
            md_file.write("# Сравнение Моделей: Генерация тестов RAG\n\n")
            
            for i in range(len(selected_examples)):
                data = md_logs[i]
                md_file.write(f"## Пример {i+1}\n")
                md_file.write(f"**Тема (XQuAD):** {data['xquad_q']}\n")
                md_file.write(f"🎯 **Эталон XQuAD:** {data['gold']}\n\n")
                
                for model_name, m_data in data["models"].items():
                    md_file.write(f"### 🤖 Модель: `{model_name}`\n")
                    md_file.write(f"**Сгенерированный тест:**\n```text\n{m_data['raw_output']}\n```\n")
                    md_file.write(f"**Прав. ответ модели:** {m_data['llm_correct_ans']}\n")
                    md_file.write(f"📊 *Метрики:* Format: {m_data['f_score']:.2f} | Sem Sim: {m_data['s_sim']:.2f} | Distr Dist: {m_data['d_dist']:.2f} | Distr Plaus: {m_data['d_plaus']:.2f} | Faithfulness: {m_data['faithfulness']:.2f}\n\n")
                md_file.write("---\n\n")
        print(f"✅ Логи (чтобы посмотреть глазами) лежат в {OUTPUT_MD}")

        print("\n" + "=" * 70)
        print("СРЕДНИЕ ЗНАЧЕНИЯ (СРАВНЕНИЕ МОДЕЛЕЙ)")
        print("=" * 70)
        
        metrics_to_agg = [
            "semantic_sim", 
            "distractor_distinctness", 
            "distractor_plausibility", 
            "format_adherence",
            "ragas_faithfulness",
            "generation_time_sec"
        ]
        
        models_found = list(set(r["model_name"] for r in results))
        for model in models_found:
            print(f"\nМодель: {model.upper()}")
            model_rows = [r for r in results if r["model_name"] == model]
            for m in metrics_to_agg:
                vals = [r[m] for r in model_rows if isinstance(r.get(m), (int, float))]
                if vals:
                    print(f"  {m:25s}: {np.mean(vals):.3f} +- {np.std(vals):.3f}")
    else:
        print("Нет результатов.")