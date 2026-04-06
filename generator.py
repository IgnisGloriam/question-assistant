# generator.py
# Шаг 7: Генерация кандидатов (Best-of-N) с помощью локальной LLM

import os
import time

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "Не установлена библиотека llama-cpp-python. "
        "Установите её командой: pip install llama-cpp-python"
    )

class LLMGenerator:
    """
    Класс для работы с локальной языковой моделью через llama.cpp.
    Загружает модель в память (с выгрузкой слоёв на GPU) и генерирует текст.
    """

    def __init__(self, model_path: str = "D:/models/qwen2.5-3b-instruct-q4_k_m.gguf"):
        """
        Инициализация модели.
        
        Args:
            model_path: Путь к файлу модели (.gguf).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена по пути: {model_path}\n"
                "Скачайте её по ссылке: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf\n"
                "и положите в папку 'models'."
            )

        print(f"Загрузка модели {model_path} в память...")
        
        # Настройки llama.cpp для 4GB VRAM и 16GB RAM + Qwen2.5-3B
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,          # Размер контекстного окна (максимум токенов на вход + выход)
            n_gpu_layers=40,     # Для 3B модели ~36 слоёв. Указываем с запасом, чтобы всё ушло на видеокарту!
            n_threads=4,         # Количество потоков процессора (если что-то не влезло в GPU)
            verbose=False        # Отключаем спам в консоль от движка C++
        )
        print("✅ Модель успешно загружена!")

    def generate_single(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Единичная генерация текста (один вариант).
        
        Args:
            prompt: Готовый текстовый запрос (из Шага 6)
            temperature: Креативность (0.1 - строго, 0.9 - разнообразно)
            
        Returns:
            Сгенерированный текст (вопросы)
        """
        # Запуск авторегрессионной генерации
        response = self.llm(
            prompt,
            max_tokens=1500,        # Ограничение на размер ответа (чтобы не генерировала бесконечно)
            temperature=temperature, # Та самая "креативность"
            top_p=0.9,              # Альтернативный параметр обрезки невероятных токенов
            stop=["<|im_end|>"],    # СТОП-СЛОВО. Очень важно! Без него модель начнёт говорить сама с собой
            echo=False              # Не возвращать сам промпт в ответе
        )
        
        # Извлекаем текст из структуры ответа llama.cpp
        return response["choices"][0]["text"].strip()

    def generate_candidates(self, prompt: str, n_candidates: int = 3) -> list[str]:
        """
        Best-of-N генерация. 
        Генерирует несколько вариантов с разной температурой.
        
        Args:
            prompt: Готовый текстовый запрос
            n_candidates: Количество вариантов (обычно 3)
            
        Returns:
            Список строк (каждая строка — набор сгенерированных вопросов)
        """
        print(f"🚀 Запуск генерации {n_candidates} кандидатов (Best-of-N)...")
        
        # Подбираем разную температуру для каждого кандидата,
        # чтобы получить разные стили и формулировки вопросов.
        # [0.3 (строго по тексту), 0.6 (баланс), 0.9 (креативно, но может ошибиться)]
        temperatures = [0.3, 0.5, 0.7, 0.4, 0.8] 
        
        candidates = []
        
        for i in range(min(n_candidates, len(temperatures))):
            current_temp = temperatures[i]
            print(f"  ⏳ Генерация варианта {i+1}/{n_candidates} (temperature={current_temp})...", end="", flush=True)
            
            start_time = time.time()
            
            # Генерируем!
            result = self.generate_single(prompt, temperature=current_temp)
            candidates.append(result)
            
            elapsed = time.time() - start_time
            print(f" Готово! ({elapsed:.1f} сек)")
            
        return candidates

# ══════════════════════════════════════
#  Запуск для проверки
# ══════════════════════════════════════

if __name__ == "__main__":

    from parser import extract_text
    from chunk import chunk_text
    from retrieval import LectureIndex, ContextRetriever

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