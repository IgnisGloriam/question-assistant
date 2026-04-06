import re
from parser import extract_text


def chunk_text(text, min_size=400, max_size=500, overlap=100):
    """
    Разбивает текст на фрагменты, стараясь резать по границам предложений.
    
    Приоритет разрезания:
      1. Конец предложения (.!?) в диапазоне [min_size, max_size]
      2. Конец предложения в расширенном диапазоне [max_size, max_size + 50]
      3. Граница слова (пробел) ближайшая к max_size
      4. Жёсткий лимит max_size (крайний случай)
    
    Args:
        text: исходный текст
        min_size: минимальный размер чанка (не ищем разрез раньше)
        max_size: желаемый максимальный размер
        overlap: перекрытие между соседними чанками
    
    Returns:
        список текстовых фрагментов
    """
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Последний фрагмент — забираем остаток
        if start + max_size >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Ищем лучшую точку разреза
        cut_point = _find_cut_point(text, start, min_size, max_size)

        chunk = text[start:cut_point].strip()
        if chunk:
            chunks.append(chunk)

        # Смещаемся с учётом перекрытия
        # Перекрытие тоже выравниваем по границе слова
        next_start = cut_point - overlap
        if next_start <= start:
            next_start = cut_point

        # Не начинаем новый чанк с середины слова
        next_start = _snap_to_word_boundary(text, next_start)

        start = next_start

    return chunks


def _find_cut_point(text, start, min_size, max_size):
    """
    Находит оптимальную точку разреза.
    
    Приоритет:
      1. Конец предложения в [min_size, max_size]
      2. Конец предложения в [max_size, max_size + 50] (небольшой допуск)
      3. Граница слова, ближайшая к max_size
      4. Жёсткий max_size
    """
    # Паттерн конца предложения:
    # точка/!/? за которыми идёт пробел, перенос строки или конец текста
    # Но НЕ точка внутри сокращений типа "т.д.", "т.е.", "др.", "рис."
    sentence_end = re.compile(
        r'(?<!\b\w)'       # не одна буква перед точкой (т., д., и.)
        r'[.!?]'           # сам знак
        r'(?:\s|$)'        # после — пробел или конец
    )

    absolute_start = start + min_size
    absolute_end = start + max_size

    # ── Приоритет 1: конец предложения в основном диапазоне ──
    search_area = text[absolute_start:absolute_end]
    matches = list(sentence_end.finditer(search_area))

    if matches:
        # Берём последнее предложение, которое влезает
        last_match = matches[-1]
        return absolute_start + last_match.end()

    # ── Приоритет 2: конец предложения чуть за пределом (допуск 50 символов) ──
    grace = 50
    extended_area = text[absolute_end:absolute_end + grace]
    matches = list(sentence_end.finditer(extended_area))

    if matches:
        # Берём первое предложение за границей — ближайшее
        first_match = matches[0]
        return absolute_end + first_match.end()

    # ── Приоритет 3: граница слова (пробел) ближайшая к max_size ──
    # Ищем последний пробел в основном диапазоне
    search_area = text[absolute_start:absolute_end]
    space_matches = list(re.finditer(r'\s+', search_area))

    if space_matches:
        last_space = space_matches[-1]
        return absolute_start + last_space.start()

    # ── Приоритет 4: жёсткий лимит (текст без пробелов — маловероятно) ──
    return absolute_end


def _snap_to_word_boundary(text, position):
    """
    Сдвигает позицию вперёд до ближайшей границы слова,
    чтобы чанк не начинался с середины слова.
    """
    if position >= len(text):
        return position

    # Если уже на пробеле — сдвигаемся за пробел
    if text[position].isspace():
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    # Если внутри слова — ищем следующий пробел
    while position < len(text) and not text[position].isspace():
        position += 1

    # Пропускаем сам пробел
    while position < len(text) and text[position].isspace():
        position += 1

    return position



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Использование: python chunk.py <путь_к_файлу>")
        print("Поддерживаемые форматы: .txt, .md, .docx, .pdf, .pptx")
        sys.exit(1)

    path = sys.argv[1]

    text = extract_text(path)
    print(f"Исходный текст: {len(text)} символов\n")

    result = chunk_text(text)

    for i, chunk in enumerate(result):
        # Показываем первое и последнее слово для проверки границ
        first_word = chunk.split()[0] if chunk.split() else ""
        last_word = chunk.split()[-1] if chunk.split() else ""

        print(f"--- Чанк {i+1} ({len(chunk)} симв.) ---")
        print(f"    Начало: «{first_word}...»")
        print(f"    Конец:  «...{last_word}»")
        print(chunk)
        print()