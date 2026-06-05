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
    context = retriever.get_context(topic=topic, n_results=5, max_context_chars=3000)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM:")
    print("=" * 60)
    print(context)



    from retrieval.my_retrieval import MyContextRetriever

    my_retriever = MyContextRetriever(index)
    my_context = my_retriever.get_context(topic=topic, n_results=20)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM (MyContextRetriever):")
    print("=" * 60)
    print(my_context)


    from retrieval.mmr_retrieval import MMRContextRetriever

    mmr_retriever = MMRContextRetriever(index)
    mmr_context = mmr_retriever.get_context(query=topic, top_m=20, n_select=10)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM (MMRContextRetriever):")
    print("=" * 60)
    print(mmr_context)



    from retrieval.dpp_retrieval import DPPContextRetriever

    dpp_retriever = DPPContextRetriever(index)
    dpp_context, selected = dpp_retriever.get_context(topic=topic, top_m=20, n_select=10)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM (DPPContextRetriever):")
    print("=" * 60)
    print(dpp_context)

    from prompt_builder import build_generation_prompt 

    test_context = context
    test_context1 = my_context
    test_context2 = mmr_context
    test_context3 = dpp_context

    prompt = build_generation_prompt(test_context, n_questions=1, difficulty="Лёгкий")
    prompt1 = build_generation_prompt(test_context1, n_questions=1, difficulty="Лёгкий")
    prompt2 = build_generation_prompt(test_context2, n_questions=1, difficulty="Лёгкий")
    prompt3 = build_generation_prompt(test_context3, n_questions=1, difficulty="Лёгкий")

    generator = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    generator1 = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    generator2 = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
    generator3 = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")

        
    candidates = generator.generate_candidates(prompt, n_candidates=1)
    candidates1 = generator1.generate_candidates(prompt1, n_candidates=1)
    candidates2 = generator2.generate_candidates(prompt2, n_candidates=1)
    candidates3 = generator3.generate_candidates(prompt3, n_candidates=1)


    for i, candidate in enumerate(candidates):
        print("\n" + "="*50)
        print(f"=== КАНДИДАТ {i+1} ===")
        print("="*50)
        print(candidate)
    
    for i, candidate in enumerate(candidates1):
        print("\n" + "="*50)
        print(f"=== КАНДИДАТ {i+1} ===")
        print("="*50)
        print(candidate)

    for i, candidate in enumerate(candidates2):
        print("\n" + "="*50)
        print(f"=== КАНДИДАТ {i+1} ===")
        print("="*50)
        print(candidate)

    for i, candidate in enumerate(candidates3):
        print("\n" + "="*50)
        print(f"=== КАНДИДАТ {i+1} ===")
        print("="*50)
        print(candidate)

        


    # prompt = build_generation_prompt(test_context, n_questions=2, difficulty="Лёгкий")
    
    # try:
        
    #     generator = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        
    #     candidates = generator.generate_candidates(prompt, n_candidates=3)
        
        
    #     for i, candidate in enumerate(candidates):
    #         print("\n" + "="*50)
    #         print(f"=== КАНДИДАТ {i+1} ===")
    #         print("="*50)
    #         print(candidate)

    # except FileNotFoundError as e:
    #     print("\n❌ ОШИБКА:")
    #     print(e)


    # print("\n" + "="*50)
    # print("\n" + "="*50)



    # try:
        
    #     llm_gen = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        
    #     best_questions = judge_and_refine(
    #         generator=llm_gen, 
    #         candidates=candidates, 
    #         context=test_context
    #     )
        
    #     print("\n🏆 === ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (Выбор судьи) ===")
    #     print(best_questions)
        
    # except FileNotFoundError as e:
    #     print("\n❌ ОШИБКА:")
    #     print(e)


