import import_module

import numpy as np
from typing import List, Tuple, Optional

from embedder import LectureIndex


def _logdet_psd(M: np.ndarray, jitter: float = 1e-8, max_tries: int = 6) -> float:
    """
    Устойчивый log(det(M)) для PSD-матриц через Холецкого.
    Если матрица почти вырождена, добавляем jitter*I.
    """
    n = M.shape[0]
    if n == 0:
        return float("-inf")
    I = np.eye(n, dtype=M.dtype)

    j = jitter
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(M + j * I)
            return float(2.0 * np.sum(np.log(np.diag(L) + 1e-30)))
        except np.linalg.LinAlgError:
            j *= 10.0

    sign, logabs = np.linalg.slogdet(M + j * I)
    return float(logabs) if sign > 0 else float("-inf")


def _quality_weights(sim_to_query: np.ndarray, alpha: float = 5.0) -> np.ndarray:
    """
    Преобразует similarity к запросу в положительные веса качества.
    exp(alpha * sim) — стандартная и устойчивая схема.
    """
    sim = np.clip(sim_to_query, -1.0, 1.0)
    w = np.exp(alpha * sim)
    return w.astype(np.float64)


def dpp_greedy_map_fixed_size(L: np.ndarray, k: int) -> List[int]:
    """
    Greedy MAP для DPP (L-ensemble) при фиксированном размере k.
    Максимизируем det(L_S).

    На малых top_m (10-30) на CPU работает быстро.
    """
    n = L.shape[0]
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []

    selected: List[int] = []
    remaining = list(range(n))

    # старт: лучший одиночный элемент по L_ii (det = L_ii)
    diag = np.clip(np.diag(L), 0.0, None)
    first = int(np.argmax(diag))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        best_i = None
        best_score = float("-inf")

        # текущий logdet
        cur_sub = L[np.ix_(selected, selected)]
        cur_logdet = _logdet_psd(cur_sub)

        for i in remaining:
            idxs = selected + [i]
            sub = L[np.ix_(idxs, idxs)]
            logdet = _logdet_psd(sub)

            gain = logdet - cur_logdet
            if gain > best_score:
                best_score = gain
                best_i = i

        if best_i is None:
            break

        selected.append(int(best_i))
        remaining.remove(int(best_i))

    return selected


def dpp_select_texts(
    index: LectureIndex,
    texts: List[str],
    query: str,
    n_select: int = 5,
    alpha: float = 5.0,
) -> List[int]:
    """
    Универсальная функция: по списку texts выбирает n_select элементов через DPP,
    используя fastembed-эмбеддинги из index.embed_texts().

    Returns: индексы выбранных элементов (в порядке выбора greedy).
    """
    if not texts:
        return []
    if len(texts) <= n_select:
        return list(range(len(texts)))

    # embeddings
    X = index.embed_texts(["passage: " + t for t in texts], normalize=True).astype(np.float64)
    q = index.embed_texts(["query: " + query], normalize=True)[0].astype(np.float64)

    # relevance
    sim_to_query = (X @ q)  # (n,)
    w = _quality_weights(sim_to_query, alpha=alpha)  # positive

    # build L = B B^T, where B_i = w_i * x_i
    B = X * w[:, None]
    L = B @ B.T  # PSD Gram

    # greedy MAP
    idx = dpp_greedy_map_fixed_size(L, k=n_select)
    return idx


class DPPContextRetriever:
    """
    Retrieval как обычно (Chroma top-m), затем выбор n_select через DPP.
    """

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
        top_m: int = 15,
        n_select: int = 5,
        alpha: float = 5.0,
        max_context_chars: int = 3000,
        preserve_candidate_order: bool = True,
    ) -> Tuple[str, List[str]]:
        """
        Returns: (context, selected_fragments)
        """
        # 1) candidates
        if topic.strip():
            candidates = self._search_by_topic(topic, top_m)
            query = topic
        else:
            candidates = self._search_broad(top_m)
            query = "основные понятия урока"

        if not candidates:
            return "", []

        # 2) DPP selection on candidates
        idx = dpp_select_texts(
            index=self.index,
            texts=candidates,
            query=query,
            n_select=n_select,
            alpha=alpha,
        )

        if preserve_candidate_order:
            idx = sorted(idx)

        selected = [candidates[i] for i in idx]
        context = self._build_context(selected, max_context_chars)

        return context, selected

    def _search_by_topic(self, topic: str, n_results: int) -> List[str]:
        print(f"Поиск по теме: «{topic}» (top_m={n_results})")
        return self.index.search(topic, n_results=n_results)

    def _search_broad(self, n_results_per_query: int) -> List[str]:
        print("Тема не указана, ищем по общим запросам (DPP)...")

        all_fragments: List[str] = []
        seen = set()

        for query in self.DEFAULT_QUERIES:
            results = self.index.search(query, n_results=n_results_per_query)
            for fragment in results:
                if fragment not in seen:
                    seen.add(fragment)
                    all_fragments.append(fragment)

        return all_fragments

    def _build_context(self, fragments: List[str], max_chars: int) -> str:
        parts = []
        total = 0
        for fr in fragments:
            if total + len(fr) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(fr[:remaining] + "...")
                elif remaining > 0:
                    parts.append(fr[:remaining])
                break
            parts.append(fr)
            total += len(fr)

        context = "\n\n".join(parts).strip()
        print(f"Контекст (DPP) собран: {len(context)} символов из {len(parts)} фрагментов")
        return context


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    from parser import extract_text
    from chunk import chunk_text

    path = "test_text/1 engl.docx"
    topic = "Bluetooth"

    text = extract_text(path)
    chunks = chunk_text(text)

    index = LectureIndex()
    index.index_chunks(chunks)

    retriever = DPPContextRetriever(index)
    context, selected = retriever.get_context(
        topic=topic,
        top_m=15,
        n_select=5,
        alpha=5.0,
        max_context_chars=3000,
        preserve_candidate_order=True,
    )

    print("\n" + "=" * 60)
    print("SELECTED FRAGMENTS (DPP):")
    print("=" * 60)
    for i, fr in enumerate(selected, 1):
        print(f"\n--- #{i} ---\n{fr[:400]}{'...' if len(fr) > 400 else ''}")

    print("\n" + "=" * 60)
    print("FINAL CONTEXT (DPP):")
    print("=" * 60)
    print(context)