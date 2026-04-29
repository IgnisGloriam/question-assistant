import import_module

from embedder import LectureIndex


class ContextRetriever:
    DEFAULT_QUERIES = [
        "основные понятия и определения",
        "ключевые идеи и выводы",
        "примеры и закономерности",
    ]

    def __init__(self, index: LectureIndex):
        self.index = index

    def get_context(
        self,
        topic: str = "",
        n_results: int = 5,
        max_context_chars: int = 3000
    ) -> str:
        if topic.strip():
            fragments = self._search_by_topic(topic, n_results)
        else:
            fragments = self._search_broad(n_results)

        context = self._build_context(fragments, max_context_chars)

        return context

    def _search_by_topic(self, topic: str, n_results: int) -> list[str]:
        print(f"Поиск по теме: «{topic}»")
        results = self.index.search(topic, n_results=n_results)
        return results

    def _search_broad(self, n_results_per_query: int) -> list[str]:
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
        context_parts = []
        total_length = 0

        for i, fragment in enumerate(fragments):
            if total_length + len(fragment) > max_chars:
                
                remaining = max_chars - total_length
                if remaining > 100:
                    context_parts.append(fragment[:remaining] + "...")
                elif remaining > 0:
                    context_parts.append(fragment)
                
                break

            context_parts.append(fragment)
            total_length += len(fragment)

        context = "\n\n".join(context_parts)

        print(f"Контекст собран: {len(context)} символов "
              f"из {len(context_parts)} фрагментов")

        return context




if __name__ == "__main__":
    # import sys
    from parser import extract_text
    from chunk import chunk_text

    path = 'test_text/1 engl.docx'
    topic = 'Bluetooth'

    # if len(sys.argv) < 2:
    #     print("Использование: python retrieval.py <файл> [тема]")
    #     sys.exit(1)

    # path = sys.argv[1]
    # topic = sys.argv[2] if len(sys.argv) > 2 else ""


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