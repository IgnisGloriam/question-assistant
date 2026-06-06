import re
from parser import extract_text


def chunk_text(text, min_size=400, max_size=500, overlap=100):
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        if start + max_size >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        cut_point = _find_cut_point(text, start, min_size, max_size)

        chunk = text[start:cut_point].strip()
        if chunk:
            chunks.append(chunk)

        next_start = cut_point - overlap
        if next_start <= start:
            next_start = cut_point

        next_start = _snap_to_word_boundary(text, next_start)

        start = next_start

    return chunks


def _find_cut_point(text, start, min_size, max_size):
    sentence_end = re.compile(
        r'(?<!\b\w)'
        r'[.!?]'
        r'(?:\s|$)'
    )

    absolute_start = start + min_size
    absolute_end = start + max_size

    search_area = text[absolute_start:absolute_end]
    matches = list(sentence_end.finditer(search_area))

    if matches:
        last_match = matches[-1]
        return absolute_start + last_match.end()

    grace = 50
    extended_area = text[absolute_end:absolute_end + grace]
    matches = list(sentence_end.finditer(extended_area))

    if matches:
        first_match = matches[0]
        return absolute_end + first_match.end()


    search_area = text[absolute_start:absolute_end]
    space_matches = list(re.finditer(r'\s+', search_area))

    if space_matches:
        last_space = space_matches[-1]
        return absolute_start + last_space.start()

    return absolute_end


def _snap_to_word_boundary(text, position):
    if position >= len(text):
        return position

    if text[position].isspace():
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    while position < len(text) and not text[position].isspace():
        position += 1

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
        first_word = chunk.split()[0] if chunk.split() else ""
        last_word = chunk.split()[-1] if chunk.split() else ""

        print(f"--- Чанк {i+1} ({len(chunk)} симв.) ---")
        print(f"    Начало: «{first_word}...»")
        print(f"    Конец:  «...{last_word}»")
        print(chunk)
        print()