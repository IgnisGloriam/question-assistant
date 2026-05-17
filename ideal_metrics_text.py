import os
import csv
import re
import math
import time
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embedder import LectureIndex
from parser import extract_text
from chunk import chunk_text
from retrieval.retrieval import ContextRetriever
from generator import LLMGenerator
from ragas_metrics import LocalRAGAS

# ============================================================
# НАСТРОЙКИ (КОНФИГУРАЦИЯ)
# ============================================================

DOC_PATH      = "test_text/pc.txt"  # Путь к твоему тексту
# Темы, по которым нужно сгенерировать тесты (т.к. у нас нет XQuAD вопросов)
TOPICS        = ["Bluetooth", "История создания сетей"]

LLM_PATH      = "D:/models/qwen2.5-3b-instruct-q4_k_m.gguf"
K_SELECT      = 3        
N_CANDIDATES  = 3

OUTPUT_CSV    = "rag_custom_doc_results.csv"
OUTPUT_MD     = "rag_custom_doc_log.md"

FIELDNAMES = [
    "topic", "generated_question", 
    "distractor_distinctness", "distractor_plausibility", "format_adherence",
    "ragas_faithfulness", "ragas_answer_relevancy",
    "generation_time_sec", "llm_correct_answer"
]

# ============================================================
# МЕТРИКИ (БЕЗ ЭТАЛОННОГО ОТВЕТА)
# ============================================================

def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("ё", "е")
    s = re.sub(r"[^\w\sа-яА-ЯёЁ\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

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
        # Самое важное: подтверждается ли ответ текстом из файла?
        "ragas_faithfulness":     float(ragas.faithfulness(pred_answer, chunks)),
        # Отвечает ли ответ на заданный вопрос?
        "ragas_answer_relevancy": float(ragas.answer_relevancy(question, pred_answer)),
    }

# ============================================================
# ПРОМПТЫ И ПАРСИНГ
# ============================================================

def build_test_prompt(context: str, topic: str) -> str:
    return f"""<|im_start|>system
Ты — строгий экзаменатор. Составь ОДИН тестовый вопрос по предоставленному тексту.
Тема вопроса: {topic}

ПРАВИЛА:
1. Придумай ровно 4 РАЗНЫХ варианта ответа.
2. В строке "Правильный ответ:" напиши САМ ТЕКСТ ответа, а не просто букву.

Формат ответа СТРОГО следующий:
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

def build_judge_prompt(candidates: List[str], context: str, topic: str) -> str:
    candidates_text = "\n\n".join([f"--- КАНДИДАТ {i+1} ---\n{c}" for i, c in enumerate(candidates)])
    return f"""<|im_start|>system
Ты — главный редактор тестов. Тебе даны несколько черновых вариантов тестового вопроса на тему "{topic}" и исходный текст.
Твоя задача: выбрать самый лучший, фактологически точный и логичный тест.
Если у лучшего кандидата есть ошибки формата, исправь их. Варианты ответов не должны повторяться.

Выведи ИТОГОВЫЙ ТЕСТ строго в таком формате:
Вопрос: [Текст вопроса]
А) [Вариант 1]
Б) [Вариант 2]
В) [Вариант 3]
Г) [Вариант 4]
Правильный ответ: [Текст правильного варианта без буквы]<|im_end|>
<|im_start|>user
ИСХОДНЫЙ ТЕКСТ:
{context}

КАНДИДАТЫ:
{candidates_text}

Выведи только финальный тест, без рассуждений:<|im_end|>
<|im_start|>assistant
"""

def parse_generated_test(raw_text: str) -> Tuple[str, List[str], List[str], str]:
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    lines = clean_text.split('\n')
    question, correct_answer = "", ""
    options = []
    
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

# ============================================================
# LLM AS A JUDGE ЛОГИКА
# ============================================================

def judge_and_refine(generator: LLMGenerator, candidates: List[str], context: str, topic: str) -> str:
    print("\n  ⚖️ Запуск LLM-судьи для выбора лучшего варианта...")
    prompt = build_judge_prompt(candidates, context, topic)
    # Используем низкую температуру для судьи, чтобы он был строгим и логичным
    final_result = generator.generate_single(prompt=prompt, temperature=0.1)
    return final_result

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Загрузка и чанкинг локального файла: {DOC_PATH}")
    text = extract_text(DOC_PATH)
    chunks = chunk_text(text, min_size=200, max_size=500, overlap=50)
    
    print("Индексация базы данных...")
    global_index = LectureIndex()
    global_index.index_chunks(chunks)

    generator = LLMGenerator(model_path=LLM_PATH)
    results = []

    with open(OUTPUT_MD, "w", encoding="utf-8") as md_file:
        md_file.write("# Анализ тестов на локальном документе (Reference-Free + Judge)\n\n")

        for t_idx, topic in enumerate(TOPICS):
            print(f"\n{'='*60}")
            print(f"Тема {t_idx+1}/{len(TOPICS)}: {topic}")
            print(f"{'='*60}")

            # 1. Поиск (Retrieval)
            retriever = ContextRetriever(global_index)
            retrieved_context = retriever.get_context(topic=topic, n_results=K_SELECT, max_context_chars=2000)
            retrieved_chunks = global_index.search(topic, n_results=K_SELECT)

            t0 = time.time()
            
            # 2. Генерация N кандидатов (Best-of-N)
            print(f"  📝 Генерация {N_CANDIDATES} кандидатов...")
            gen_prompt = build_test_prompt(retrieved_context, topic)
            candidates = generator.generate_candidates(gen_prompt, n_candidates=N_CANDIDATES)

            # 3. Суд (LLM-as-a-Judge)
            final_raw_output = judge_and_refine(generator, candidates, retrieved_context, topic)
            
            elapsed = time.time() - t0

            # 4. Парсинг итогового теста
            gen_q, all_options, distractors, llm_correct_ans = parse_generated_test(final_raw_output)

            # 5. Метрики без эталона
            d_dist = distractor_distinctness(llm_correct_ans, distractors, global_index)
            d_plaus = distractor_plausibility(gen_q, distractors, global_index)
            f_score = format_adherence_score(gen_q, all_options, llm_correct_ans)
            
            rgs = compute_ragas(gen_q, llm_correct_ans, retrieved_chunks, global_index)

            # Логирование
            md_file.write(f"## Тема: {topic}\n\n")
            md_file.write(f"**Найденный контекст (выжимка):**\n> {retrieved_context[:300].replace(chr(10), ' ')}...\n\n")
            md_file.write(f"🏆 **Итоговый тест (После Судьи):**\n```text\n{final_raw_output.strip()}\n```\n\n")
            md_file.write(f"📊 *Метрики:* Format Score: {f_score:.2f} | Distr Distinctness: {d_dist:.2f} | Distr Plausibility: {d_plaus:.2f} | Faithfulness: {rgs['ragas_faithfulness']:.2f}\n")
            md_file.write("---\n\n")

            # Сбор данных для CSV
            results.append({
                "topic": topic,
                "generated_question": gen_q,
                "distractor_distinctness": d_dist,
                "distractor_plausibility": d_plaus,
                "format_adherence": f_score,
                "ragas_faithfulness": rgs["ragas_faithfulness"],
                "ragas_answer_relevancy": rgs["ragas_answer_relevancy"],
                "generation_time_sec": elapsed,
                "llm_correct_answer": llm_correct_ans
            })
            
            print(f"  ✅ Готово за {elapsed:.1f}s | Формат: {f_score:.2f} | Faithfulness: {rgs['ragas_faithfulness']:.2f}")

    # Сохранение и вывод
    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"\n✅ Результаты сохранены в {OUTPUT_CSV}")
        print(f"✅ Логи (чтобы посмотреть глазами) лежат в {OUTPUT_MD}")

        print("\n" + "=" * 65)
        print("СРЕДНИЕ ЗНАЧЕНИЯ КАЧЕСТВА ТЕСТОВ (Reference-Free)")
        print("=" * 65)
        
        metrics_to_agg = [
            "distractor_distinctness", 
            "distractor_plausibility", 
            "format_adherence",
            "ragas_faithfulness",
            "ragas_answer_relevancy"
        ]
        for m in metrics_to_agg:
            vals = [r[m] for r in results if isinstance(r.get(m), (int, float))]
            if vals:
                print(f"  {m:25s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    del generator