from embedder import LectureIndex


class ContextRetriever:
    """
    Получает из векторной базы наиболее релевантные фрагменты
    и склеивает их в готовый контекст для языковой модели.
    """

    # Запросы по умолчанию, если учитель не указал тему
    DEFAULT_QUERIES = [
        "основные понятия и определения",
        "ключевые идеи и выводы",
        "примеры и закономерности",
    ]

    def __init__(self, index: LectureIndex):
        """
        Args:
            index: проиндексированная база фрагментов (из embedder.py)
        """
        self.index = index

    def get_context(
        self,
        topic: str = "",
        n_results: int = 5,
        max_context_chars: int = 3000
    ) -> str:
        """
        Находит релевантные фрагменты и собирает контекст.

        Args:
            topic: тема от учителя (может быть пустой)
            n_results: сколько фрагментов искать на один запрос
            max_context_chars: ограничение длины итогового контекста

        Returns:
            склеенный текст из найденных фрагментов
        """
        if topic.strip():
            # Учитель указал тему — ищем по ней
            fragments = self._search_by_topic(topic, n_results)
        else:
            # Тема не указана — ищем по нескольким общим запросам
            fragments = self._search_broad(n_results)

        # Собираем контекст с учётом лимита символов
        context = self._build_context(fragments, max_context_chars)

        return context

    def _search_by_topic(self, topic: str, n_results: int) -> list[str]:
        """Поиск по конкретной теме"""
        print(f"Поиск по теме: «{topic}»")
        results = self.index.search(topic, n_results=n_results)
        return results

    def _search_broad(self, n_results_per_query: int) -> list[str]:
        """
        Поиск по нескольким общим запросам.
        Убирает дубли, сохраняя порядок.
        """
        print("Тема не указана, ищем по общим запросам...")

        all_fragments = []
        seen = set()

        for query in self.DEFAULT_QUERIES:
            results = self.index.search(
                query,
                n_results=n_results_per_query
            )
            for fragment in results:
                if fragment not in seen:
                    seen.add(fragment)
                    all_fragments.append(fragment)

        return all_fragments

    def _build_context(
        self,
        fragments: list[str],
        max_chars: int
    ) -> str:
        """
        Склеивает фрагменты в один текст.
        Обрезает, если суммарная длина превышает лимит.
        """
        context_parts = []
        total_length = 0

        for i, fragment in enumerate(fragments):
            if total_length + len(fragment) > max_chars:
                # Добавляем остаток, который влезает
                remaining = max_chars - total_length
                if remaining > 100:
                    context_parts.append(fragment[:remaining] + "...")
                break

            context_parts.append(fragment)
            total_length += len(fragment)

        context = "\n\n".join(context_parts)

        print(f"Контекст собран: {len(context)} символов "
              f"из {len(context_parts)} фрагментов")

        return context


# ══════════════════════════════════════
#  Запуск для проверки
# ══════════════════════════════════════

if __name__ == "__main__":
    import sys
    from parser import extract_text
    from chunk import chunk_text

    path = 'test_text/1 engl.docx'
    topic = 'Bluetooth'

    # if len(sys.argv) < 2:
    #     print("Использование: python retrieval.py <файл> [тема]")
    #     sys.exit(1)

    # path = sys.argv[1]
    # topic = sys.argv[2] if len(sys.argv) > 2 else ""

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