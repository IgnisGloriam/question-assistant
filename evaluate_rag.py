import os
import csv
import time
import numpy as np
from datasets import load_dataset
import gc

from embedder import LectureIndex
from chunk import chunk_text
from retrieval import ContextRetriever
from generator import LLMGenerator
from ideal_metric import parse_generated_test, rouge_l, bleu_4, semantic_similarity, distractor_distinctness, distractor_plausibility, format_adherence_score, compute_ragas


N_XQUAD_SAMPLES = 150
K_SELECT = 3
OUTPUT_CSV = "results_xquad.csv"
OUTPUT_MD = "results_log.md"

LLM_PATHS = [
    "D:/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    "D:/models/qwen3.5-2b-q4_k_m.gguf",
    "D:/models/Qwen3.5-0.8B-Q4_K_M.gguf",
    "D:/models/Qwen3-1.7B-Q4_K_M-Instruct.gguf",
]

FIELDNAMES = [
    "model_name", "example_id", "xquad_original_question", "generated_question", 
    "rouge_l", "bleu4", "semantic_sim", "distractor_distinctness", 
    "distractor_plausibility", "format_adherence", "ragas_faithfulness", 
    "ragas_answer_relevancy", "generation_time_sec", "xquad_gold_answer", "llm_correct_answer"
]

def build_test_prompt(context, topic):
    return f"""<|im_start|>system
Ты — опытный методист и помощник преподавателя. Твоя задача — составлять качественные тестовые вопросы строго на основе предоставленного учебного материала.
Составь ОДИН тестовый вопрос по предоставленному тексту. Не придумывай факты, которых нет в тексте.
Тема вопроса: {topic}
Постарайся сделать так, чтобы вопрос был "{topic}", если это возможно. Если нет, то как можно меньше меняй его.

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

if __name__ == "__main__":
    print("Загрузка датасета XQuAD (ru)...")
    ds = load_dataset("xquad", "xquad.ru")["validation"]
    seed = 42
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
        print(f"\n{'='*60}\n🚀 ИНИЦИАЛИЗАЦИЯ МОДЕЛИ: {model_name}\n{'='*60}")
        
        generator = LLMGenerator(model_path=model_path)
        
        for i, ex in enumerate(selected_examples):
            context_full = ex["context"]
            xquad_q = ex["question"]
            gold_answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

            print(f"[{i+1}/{n_limit}] Тема: {xquad_q}")
            
            chunks = chunk_text(context_full, min_size=200, max_size=500, overlap=50)
            current_index = LectureIndex()
            current_index.index_chunks(chunks)

            retriever = ContextRetriever(current_index)
            retrieved_context = retriever.get_context(topic=xquad_q, n_results=K_SELECT, max_context_chars=2000)
            retrieved_chunks = current_index.search(xquad_q, n_results=K_SELECT)

            t0 = time.time()
            prompt = build_test_prompt(retrieved_context, topic=xquad_q)

            if ("qwen3.5" in model_path):
                prompt += "<think> </think>"

            raw_output = generator.generate_candidates(prompt, n_candidates=1, temperatures=[0.3])[0]
            elapsed = time.time() - t0

            gen_q, all_options, distractors, llm_correct_ans = parse_generated_test(raw_output)

            r_l = rouge_l(llm_correct_ans, gold_answer)
            b_4 = bleu_4(llm_correct_ans, gold_answer)
            s_sim = semantic_similarity(llm_correct_ans, gold_answer, current_index)
            d_dist = distractor_distinctness(llm_correct_ans, distractors, current_index)
            d_plaus = distractor_plausibility(gen_q, distractors, current_index) 
            f_score = format_adherence_score(gen_q, all_options, llm_correct_ans) 
            rgs = compute_ragas(gen_q, llm_correct_ans, retrieved_chunks, current_index)

            print(f"   Формат: {f_score:.2f} | SemSim: {s_sim:.2f} | Время: {elapsed:.1f}s")

            results.append({
                "model_name": model_name, "example_id": ex["id"], "xquad_original_question": xquad_q,
                "generated_question": gen_q, "rouge_l": r_l, "bleu4": b_4, "semantic_sim": s_sim,
                "distractor_distinctness": d_dist, "distractor_plausibility": d_plaus,
                "format_adherence": f_score, "ragas_faithfulness": rgs["ragas_faithfulness"],
                "ragas_answer_relevancy": rgs["ragas_answer_relevancy"], "generation_time_sec": elapsed,
                "xquad_gold_answer": gold_answer, "llm_correct_answer": llm_correct_ans
            })

        del generator
        gc.collect()

    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Результаты сохранены в {OUTPUT_CSV}")