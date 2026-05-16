try:
    import import_module
except:
    pass


from typing import Tuple, List

from embedder import LectureIndex
from retrieval import ContextRetriever
from context_compressor import ContextCompressor


class CompressedContextRetriever:
    def __init__(self, index: LectureIndex):
        self.index = index
        self.base_retriever = ContextRetriever(index)
        self.compressor = ContextCompressor(index)

    def get_context(
        self,
        topic: str = "",
        n_results: int = 5,
        top_m: int = 15,  # не используется, но нужен для совместимости API
        max_context_chars: int = 3000,
        # параметры компрессии
        sim_threshold: float = 0.92,
        n_projections: int = 4,
        neighbor_window: int = 2,
        min_sentence_chars: int = 25,
        **kwargs  # для совместимости с другими retriever'ами
    ) -> Tuple[str, List[str]]:
        """
        Returns:
            context: str — сжатый контекст
            selected_fragments: List[str] — исходные чанки (до сжатия)
                                            для честных retrieval-метрик
        """
        # Шаг 1: получаем raw context через базовый retriever
        raw_context = self.base_retriever.get_context(
            topic=topic,
            n_results=n_results,
            max_context_chars=max_context_chars
        )

        # Шаг 2: сжимаем через ContextCompressor
        compressed_context, info = self.compressor.compress_context(
            text=raw_context,
            query=topic,
            sim_threshold=sim_threshold,
            n_projections=n_projections,
            neighbor_window=neighbor_window,
            min_sentence_chars=min_sentence_chars,
            max_output_chars=max_context_chars,
        )

        # Для retrieval-метрик возвращаем исходные чанки (до сжатия)
        # чтобы считать diversity/coverage/MRR корректно
        selected_fragments = self.index.search(topic, n_results=n_results)

        return compressed_context, selected_fragments


if __name__ == "__main__":
    from parser import extract_text
    from chunk import chunk_text

    path = "test_text/1 engl.docx"
    topic = "Bluetooth"

    text = extract_text(path)
    chunks = chunk_text(text)

    index = LectureIndex()
    index.index_chunks(chunks)

    # Используем как обычный retriever
    retriever = CompressedContextRetriever(index)
    context, selected = retriever.get_context(topic=topic, n_results=5)

    print(f"Выбрано фрагментов: {len(selected)}")
    print("\n" + "=" * 60)
    print("COMPRESSED CONTEXT:")
    print("=" * 60)
    print(context)