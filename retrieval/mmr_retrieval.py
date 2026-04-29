import import_module

import numpy as np
from typing import List, Tuple
from embedder import LectureIndex


def mmr_select(
    X: np.ndarray,
    q: np.ndarray,
    n_select: int = 5,
    lambda_mult: float = 0.6,
) -> List[int]:
    """
    MMR selection на нормированных эмбеддингах.
    X: (n, d) L2-normalized
    q: (d,)   L2-normalized
    Returns: индексы выбранных элементов
    """
    n = X.shape[0]
    n_select = min(n_select, n)

    sim_to_query = X @ q  # cosine similarity
    selected = [int(np.argmax(sim_to_query))]

    candidates = set(range(n))
    candidates.remove(selected[0])

    # предварительно считаем матрицу попарных сходств, чтобы не пересчитывать в цикле
    sim_mat = X @ X.T

    while len(selected) < n_select and candidates:
        best_idx = None
        best_score = -1e9

        for i in candidates:
            redundancy = np.max(sim_mat[i, selected])  # max sim to selected set
            score = lambda_mult * sim_to_query[i] - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(int(best_idx))
        candidates.remove(best_idx)

    return selected


class MMRContextRetriever:
    def __init__(self, index):
        self.index = index

    def get_context_mmr(
        self,
        query: str,
        top_m: int = 10,
        n_select: int = 5,
        lambda_mult: float = 0.6,
        max_context_chars: int = 3000,
    ) -> Tuple[str, List[str]]:
        # 1) initial retrieval by relevance (Chroma cosine)
        candidates = self.index.search(query, n_results=top_m)
        if not candidates:
            return "", []

        if len(candidates) <= n_select:
            return self._join_with_limit(candidates, max_context_chars), candidates

        # 2) embed candidates and query (fastembed inside index)
        X = self.index.embed_texts(
            ["passage: " + c for c in candidates],
            normalize=True
        ).astype(np.float64)

        q = self.index.embed_texts(
            ["query: " + query],
            normalize=True
        )[0].astype(np.float64)

        # 3) MMR selection
        idx = mmr_select(X, q, n_select=n_select, lambda_mult=lambda_mult)
        selected = [candidates[i] for i in idx]

        # 4) build context
        context = self._join_with_limit(selected, max_context_chars)
        return context, selected

    @staticmethod
    def _join_with_limit(fragments: List[str], max_chars: int) -> str:
        parts, total = [], 0
        for fr in fragments:
            if total + len(fr) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(fr[:remaining] + "...")
                break
            parts.append(fr)
            total += len(fr)
        return "\n\n".join(parts)


if __name__ == "__main__":
    # import sys
    from parser import extract_text
    from chunk import chunk_text

    path = 'test_text/med.pdf'
    topic = 'RAG limitations'


    text = extract_text(path)
    chunks = chunk_text(text)
    index = LectureIndex()
    index.index_chunks(chunks)


    retriever = MMRContextRetriever(index)
    context, selected_fragments = retriever.get_context_mmr(query=topic, n_select=5)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM:")
    print("=" * 60)
    print(context)