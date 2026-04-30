try:
    import import_module
except:
    pass

import numpy as np
from typing import List, Tuple, Optional
from embedder import LectureIndex


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Косинусная матрица сходств для уже L2-нормированных векторов."""
    return X @ X.T


def build_affinity(W_cos: np.ndarray, mode: str = "cosine_clip", sigma: float = 0.5) -> np.ndarray:
    """
    Превращает косинусные сходства в матрицу смежности (affinity) W.
    mode:
      - cosine_clip: max(0, cos)
      - rbf: exp(- (1-cos)^2 / (2 sigma^2))
    """
    if mode == "cosine_clip":
        W = np.maximum(0.0, W_cos)
        np.fill_diagonal(W, 0.0)
        return W

    if mode == "rbf":
        dist = 1.0 - W_cos
        W = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        np.fill_diagonal(W, 0.0)
        return W

    raise ValueError(f"Unknown affinity mode: {mode}")


def normalized_laplacian(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L_sym = I - D^{-1/2} W D^{-1/2}"""
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + eps))
    I = np.eye(W.shape[0])
    return I - D_inv_sqrt @ W @ D_inv_sqrt


def choose_k_by_eigengap(evals: np.ndarray, k_min: int = 2, k_max: int = 6) -> int:
    """
    Eigengap heuristic по отсортированным собственным значениям лапласиана.
    Берём k в [k_min, k_max], где (λ_{k} - λ_{k-1}) максимален
    (если λ_0 ~ 0 тривиальный).
    """
    n = len(evals)
    k_max = min(k_max, n - 1)
    k_min = min(k_min, k_max)

    # evals предполагаются отсортированными по возрастанию
    gaps = []
    ks = []
    for k in range(k_min, k_max + 1):
        gap = evals[k] - evals[k - 1]
        gaps.append(gap)
        ks.append(k)
    return ks[int(np.argmax(gaps))] if gaps else 2


def kmeans(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 42) -> np.ndarray:
    """Минималистичный k-means (чтобы не тянуть sklearn)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # init: случайные центры из точек
    centers = X[rng.choice(n, size=k, replace=False)]

    for _ in range(n_iter):
        # assign
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)

        # update
        new_centers = np.vstack([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels


class SpectralContextRetriever:
    """
    Делает:
      1) берёт top_m кандидатов из Chroma (по запросу)
      2) строит граф сходств между кандидатами по их эмбеддингам
      3) spectral clustering
      4) выбирает rep из каждого кластера (макс. по сходству к запросу)
      5) возвращает контекст
    """

    def __init__(self, index):
        self.index = index

    def get_context_spectral(
        self,
        query: str,
        top_m: int = 10,
        n_select: int = 5,
        affinity_mode: str = "cosine_clip",
        sigma: float = 0.5,
        k_clusters: Optional[int] = 5,
        max_context_chars: int = 3000,
    ) -> Tuple[str, List[str]]:
        """
        Returns:
          context_text, selected_fragments
        """

        # 1) initial retrieval
        candidates = self.index.search(query, n_results=top_m)
        if not candidates:
            return "", []

        if len(candidates) <= n_select:
            return self._join_with_limit(candidates, max_context_chars), candidates

        X = self.index.embed_texts(["passage: " + c for c in candidates], normalize=True).astype(np.float64)
        q = self.index.embed_texts(["query: " + query], normalize=True)[0].astype(np.float64)

        n = X.shape[0]

        # 3) build affinity graph
        W_cos = cosine_matrix(X)
        W = build_affinity(W_cos, mode=affinity_mode, sigma=sigma)

        # 4) Laplacian + eigenvectors
        L = normalized_laplacian(W)
        evals, evecs = np.linalg.eigh(L)  # симметричная матрица => eigh

        # 5) choose k clusters
        if k_clusters is None:
            k = choose_k_by_eigengap(evals, k_min=2, k_max=min(6, n - 1))
        else:
            k = max(2, min(int(k_clusters), n - 1))

        # берём k собственных векторов, начиная со 2-го (пропускаем тривиальный)
        U = evecs[:, 1:k+1]

        # row-normalize (стандартный шаг в spectral clustering)
        U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

        labels = kmeans(U_norm, k=k, n_iter=80)

        # 6) select representatives: max cos to query in each cluster
        sims_to_query = X @ q  # так как нормировано, это cos similarity

        selected_idx = []
        for cluster_id in range(k):
            idxs = np.where(labels == cluster_id)[0]
            if len(idxs) == 0:
                continue
            best = idxs[np.argmax(sims_to_query[idxs])]
            selected_idx.append(best)

        # если кластеров получилось больше, чем нужно — добираем по общей релевантности
        # если меньше — тоже добираем
        selected_idx = list(set(selected_idx))
        selected_idx.sort(key=lambda i: sims_to_query[i], reverse=True)

        # if not enough, fill by relevance
        if len(selected_idx) < n_select:
            # добираем из оставшихся по релевантности
            rest = [i for i in range(n) if i not in selected_idx]
            rest.sort(key=lambda i: sims_to_query[i], reverse=True)
            selected_idx += rest[:(n_select - len(selected_idx))]

        selected_idx = selected_idx[:n_select]
        selected_fragments = [candidates[i] for i in selected_idx]

        context = self._join_with_limit(selected_fragments, max_context_chars)
        return context, selected_fragments

    @staticmethod
    def _join_with_limit(fragments: List[str], max_chars: int) -> str:
        parts = []
        total = 0
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


    retriever = SpectralContextRetriever(index)
    context, selected_fragments = retriever.get_context_spectral(query=topic, n_select=5)

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM:")
    print("=" * 60)
    print(context)