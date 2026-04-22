import time
from prompt_builder import build_judge_prompt

def judge_and_refine(generator, candidates: list[str], context: str) -> str:
    print("\n⚖️ Запуск LLM-судьи для выбора лучшего варианта...")
    start_time = time.time()
    
    prompt = build_judge_prompt(candidates, context)

    print("  ⏳ Анализ кандидатов и формирование финального ответа...", end="", flush=True)
    
    final_result = generator.generate_single(
        prompt=prompt,
        temperature=0.1
    )
    
    elapsed = time.time() - start_time
    print(f" Готово! ({elapsed:.1f} сек)")
    
    return final_result


if __name__ == "__main__":

    from parser import extract_text
    from chunk import chunk_text
    from retrieval import LectureIndex, ContextRetriever
    from generator import LLMGenerator

    path = 'test_text/1 engl.docx'
    topic = 'Bluetooth'


    text = extract_text(path)
    chunks = chunk_text(text)
    index = LectureIndex()
    index.index_chunks(chunks)


    retriever = ContextRetriever(index)
    context = retriever.get_context(topic=topic)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM:")
    print("=" * 60)
    print(context)

    from prompt_builder import build_generation_prompt 

    test_context = context


    prompt = build_generation_prompt(test_context, n_questions=2, difficulty="Лёгкий")
    
    try:
        
        generator = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        
        candidates = generator.generate_candidates(prompt, n_candidates=3)
        
        
        for i, candidate in enumerate(candidates):
            print("\n" + "="*50)
            print(f"=== КАНДИДАТ {i+1} ===")
            print("="*50)
            print(candidate)

    except FileNotFoundError as e:
        print("\n❌ ОШИБКА:")
        print(e)


    print("\n" + "="*50)
    print("\n" + "="*50)



    try:
        
        llm_gen = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        
        best_questions = judge_and_refine(
            generator=llm_gen, 
            candidates=candidates, 
            context=test_context
        )
        
        print("\n🏆 === ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (Выбор судьи) ===")
        print(best_questions)
        
    except FileNotFoundError as e:
        print("\n❌ ОШИБКА:")
        print(e)


