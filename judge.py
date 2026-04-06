# judge.py
# Шаг 8: Оценка и выбор лучшего кандидата (LLM-as-Judge)

import time
from prompt_builder import build_judge_prompt

def judge_and_refine(generator, candidates: list[str], context: str) -> str:
    """
    Выступает в роли строгого судьи. 
    Анализирует сгенерированные варианты вопросов, выбирает лучший и исправляет его недочёты.
    
    Args:
        generator: Экземпляр LLMGenerator (из Шага 7), в который уже загружена модель.
        candidates: Список строк, где каждая строка — набор вопросов от модели.
        context: Исходный учебный материал (фрагменты текста).
        
    Returns:
        Финальный, отшлифованный набор вопросов.
    """
    print("\n⚖️ Запуск LLM-судьи для выбора лучшего варианта...")
    start_time = time.time()
    
    # 1. Формируем специальный промпт для судьи (из Шага 6)
    prompt = build_judge_prompt(candidates, context)

    #print(prompt)
    
    # 2. Генерируем ответ. 
    # ВАЖНО: temperature=0.1. Судья не должен фантазировать. 
    # Ему нужна максимальная логика, точность и предсказуемость.
    print("  ⏳ Анализ кандидатов и формирование финального ответа...", end="", flush=True)
    
    final_result = generator.generate_single(
        prompt=prompt,
        temperature=0.1
    )
    
    elapsed = time.time() - start_time
    print(f" Готово! ({elapsed:.1f} сек)")
    
    return final_result


# ══════════════════════════════════════
#  Запуск для проверки (Имитация полного цикла 6 -> 7 -> 8)
# ══════════════════════════════════════

if __name__ == "__main__":

    from parser import extract_text
    from chunk import chunk_text
    from retrieval import LectureIndex, ContextRetriever
    from generator import LLMGenerator

    path = 'test_text/1 engl.docx'
    topic = 'Bluetooth'

    # Шаги 1-3: парсинг → чанкинг → индексация
    text = extract_text(path)
    chunks = chunk_text(text)
    index = LectureIndex()
    index.index_chunks(chunks)

    # Шаг 5: получаем контекст
    retriever = ContextRetriever(index)
    context = retriever.get_context(topic=topic)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM:")
    print("=" * 60)
    print(context)

    from prompt_builder import build_generation_prompt # Импортируем наш Шаг 6

    test_context = context

    # 1. Формируем промпт
    prompt = build_generation_prompt(test_context, n_questions=2, difficulty="Лёгкий")
    
    try:
        # 2. Инициализируем генератор (Убедись, что скачал модель в папку models!)
        generator = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        # 3. Генерируем 3 варианта кандидатов
        candidates = generator.generate_candidates(prompt, n_candidates=3)
        
        # 4. Выводим результат
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


    # ── ШАГ 8: Работа судьи ──
    try:
        # Загружаем модель (загрузится только один раз!)
        llm_gen = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        # Запускаем судью
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



if __name__ == "__main__2":
    from prompt_builder import build_generation_prompt
    
    # Пытаемся импортировать генератор. 
    # Убедитесь, что файл generator.py лежит в той же папке.
    try:
        from generator import LLMGenerator
    except ImportError:
        print("❌ ОШИБКА: Не найден файл generator.py или не установлена llama-cpp-python.")
        exit(1)

    # ── ИСХОДНЫЕ ДАННЫЕ ──
    test_context = (
        "Эволюция — это естественный процесс развития живой природы, сопровождающийся "
        "изменением генетического состава популяций, формированием адаптаций, "
        "видообразованием и вымиранием видов. Основной движущей силой эволюции "
        "Чарльз Дарвин назвал естественный отбор."
    )
    
    # ── ИМИТАЦИЯ ШАГА 7 (Генерация кандидатов) ──
    # Для экономии времени в тесте мы не будем генерировать их нейросетью заново,
    # а создадим три "фейковых" кандидата с разными ошибками, чтобы проверить, как судья их исправит.
    
    fake_candidates = [
        # Кандидат 1: Слабые дистракторы (неправильные варианты слишком очевидны)
        "**Вопрос 1:** Что назвал Чарльз Дарвин основной движущей силой эволюции?\n"
        "А) Естественный отбор\n"
        "Б) Компьютерные игры\n"
        "В) Поедание пиццы\n"
        "Г) Сон\n"
        "✅ Правильный ответ: А",
        
        # Кандидат 2: Фактическая ошибка в правильном ответе (согласно тексту)
        "**Вопрос 1:** Что является основной движущей силой эволюции по Дарвину?\n"
        "А) Искусственный отбор\n"
        "Б) Видообразование\n"
        "В) Генетический состав\n"
        "Г) Мутации\n"
        "✅ Правильный ответ: А",
        
        # Кандидат 3: Хороший вопрос, но есть "Все вышеперечисленное" (что запрещено в правилах)
        "**Вопрос 1:** Что из перечисленного является характеристикой эволюции?\n"
        "А) Изменение генетического состава популяций\n"
        "Б) Формирование адаптаций\n"
        "В) Вымирание видов\n"
        "Г) Все вышеперечисленное\n"
        "✅ Правильный ответ: Г"
    ]
    
    print("=== ИСХОДНЫЕ КАНДИДАТЫ ===")
    for i, c in enumerate(fake_candidates):
        print(f"\nКандидат {i+1}:")
        print(c)
    
    print("\n" + "="*50)
    
    # ── ШАГ 8: Работа судьи ──
    try:
        # Загружаем модель (загрузится только один раз!)
        llm_gen = LLMGenerator(model_path="D:/models/qwen2.5-3b-instruct-q4_k_m.gguf")
        
        # Запускаем судью
        best_questions = judge_and_refine(
            generator=llm_gen, 
            candidates=fake_candidates, 
            context=test_context
        )
        
        print("\n🏆 === ФИНАЛЬНЫЙ РЕЗУЛЬТАТ (Выбор судьи) ===")
        print(best_questions)
        
    except FileNotFoundError as e:
        print("\n❌ ОШИБКА:")
        print(e)