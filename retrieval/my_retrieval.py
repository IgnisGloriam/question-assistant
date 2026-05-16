try:
    import import_module
except:
    pass

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

from embedder import LectureIndex
from retrieval import ContextRetriever


def split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_into_sentences(paragraph: str) -> List[str]:
    """
    Лучше поставить razdel:
        pip install razdel
    Иначе будет fallback на простую регулярку.
    """
    try:
        from razdel import sentenize  # type: ignore
        return [s.text.strip() for s in sentenize(paragraph) if s.text.strip()]
    except Exception:
        # fallback: грубое разбиение
        raw = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        return [s.strip() for s in raw if s.strip()]


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


@dataclass
class SentenceRec:
    paragraph_id: int
    sent_id: int  # глобальный порядок предложения в контексте
    text: str


class ContextCompressor:
    """
    Принимает на вход контекст (текст), полученный от ContextRetriever,
    и производит семантическую дедупликацию предложений с сохранением структуры.
    """

    def __init__(self, index: LectureIndex):
        self.index = index

    def compress_context(
        self,
        text: str,
        query: str = "",
        sim_threshold: float = 0.92,
        n_projections: int = 4,
        neighbor_window: int = 2,
        min_sentence_chars: int = 25,
        max_output_chars: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[str, Dict]:
        """
        Приближённая дедупликация предложений:
          1. Разбиваем текст на абзацы и предложения
          2. Векторизуем предложения (эмбеддинги)
          3. Запоминаем исходный порядок
          4. Используем случайные проекции для сортировки
          5. Находим похожие предложения среди соседей (union-find)
          6. Выбираем представителя из каждой группы дублей
          7. Восстанавливаем исходный порядок и структуру абзацев
        """

        paragraphs = split_into_paragraphs(text)

        # 1. Собираем предложения с метаданными (номер абзаца, позиция)
        sents: List[SentenceRec] = []
        sent_id = 0
        for pid, par in enumerate(paragraphs):
            for st in split_into_sentences(par):
                st = st.strip()
                if st:
                    sents.append(SentenceRec(pid, sent_id, st))
                    sent_id += 1

        if not sents:
            return "", {"n_sentences": 0, "kept_sentences": 0, "removed_sentences": 0}

        sent_texts = [s.text for s in sents]

        # 2. Векторизуем предложения
        X = self.index.embed_texts(
            ["passage: " + t for t in sent_texts],
            normalize=True
        ).astype(np.float32)

        # Фильтр: слишком короткие предложения не участвуют в дедупликации
        eligible = np.array([len(t) >= min_sentence_chars for t in sent_texts], dtype=bool)

        # 3. Инициализируем Union-Find для группировки дублей
        uf = UnionFind(len(sent_texts))
        rng = np.random.default_rng(seed)
        d = X.shape[1]

        # 4-5. Случайные проекции: сортируем по проекции и сравниваем соседей
        for _ in range(n_projections):
            # Случайный вектор для проекции
            r = rng.normal(size=(d,)).astype(np.float32)
            r /= (np.linalg.norm(r) + 1e-12)

            # Проекция всех эмбеддингов на случайный вектор
            proj = X @ r
            order = np.argsort(proj)  # сортируем по проекции

            # Сравниваем соседей в отсортированном списке
            for pos in range(len(order)):
                i = int(order[pos])
                if not eligible[i]:
                    continue

                for step in range(1, neighbor_window + 1):
                    if pos + step >= len(order):
                        break
                    j = int(order[pos + step])
                    if not eligible[j]:
                        continue

                    sim = float(X[i] @ X[j])  # cosine similarity (X normalized)
                    if sim >= sim_threshold:
                        uf.union(i, j)  # объединяем в одну группу

        # 6. Собираем группы дублей
        groups: Dict[int, List[int]] = {}
        for i in range(len(sent_texts)):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        # Вычисляем релевантность к query (если задан)
        sim_to_query = None
        if query.strip():
            q = self.index.embed_texts(["query: " + query], normalize=True)[0].astype(np.float32)
            sim_to_query = (X @ q)

        # Выбираем представителя из каждой группы
        keep_idx = set()
        for root, idxs in groups.items():
            if len(idxs) == 1:
                keep_idx.add(idxs[0])
                continue

            # Выбираем наиболее релевантное к запросу или ближайшее к центроиду
            if sim_to_query is not None:
                best = max(idxs, key=lambda i: float(sim_to_query[i]))
            else:
                mean_vec = X[idxs].mean(axis=0)
                mean_vec /= (np.linalg.norm(mean_vec) + 1e-12)
                best = max(idxs, key=lambda i: float(X[i] @ mean_vec))

            keep_idx.add(best)

        # 7. Восстанавливаем исходный порядок и структуру абзацев
        kept = [s for i, s in enumerate(sents) if i in keep_idx]
        kept.sort(key=lambda s: s.sent_id)  # восстанавливаем порядок

        # Группируем по абзацам
        by_par: Dict[int, List[str]] = {pid: [] for pid in range(len(paragraphs))}
        for s in kept:
            by_par[s.paragraph_id].append(s.text)

        # Собираем абзацы обратно
        out_pars = []
        for pid in range(len(paragraphs)):
            if by_par[pid]:
                out_pars.append(" ".join(by_par[pid]).strip())

        out_text = "\n\n".join(out_pars).strip()

        # Обрезаем если нужно
        if max_output_chars is not None and len(out_text) > max_output_chars:
            out_text = out_text[:max_output_chars].rstrip() + "..."

        info = {
            "n_paragraphs": len(paragraphs),
            "n_sentences": len(sent_texts),
            "n_groups": len(groups),
            "kept_sentences": len(keep_idx),
            "removed_sentences": len(sent_texts) - len(keep_idx),
            "sim_threshold": sim_threshold,
            "n_projections": n_projections,
            "neighbor_window": neighbor_window,
        }
        return out_text, info


if __name__ == "__main__":
    from parser import extract_text
    from chunk import chunk_text

    path = "test_text/1 engl.docx"
    topic = "Bluetooth"

    text = extract_text(path)
    chunks = chunk_text(text)

    index = LectureIndex()
    index.index_chunks(chunks)

    # Используем базовый ContextRetriever для получения исходного контекста
    retriever = ContextRetriever(index)
    raw_context = retriever.get_context(topic=topic, n_results=5, max_context_chars=3000)

    print("\n" + "=" * 60)
    print("RAW CONTEXT:")
    print("=" * 60)
    print(raw_context)

    # Применяем сжатие контекста
    compressor = ContextCompressor(index)
    compressed_context, info = compressor.compress_context(
        text=raw_context,
        query=topic,
        sim_threshold=0.92,
        n_projections=4,
        neighbor_window=2,
        min_sentence_chars=25,
        max_output_chars=3000,
    )

    print(
        f"\nДедупликация: было {info['n_sentences']}, стало {info['kept_sentences']} "
        f"(удалено {info['removed_sentences']})"
    )

    print("\n" + "=" * 60)
    print("COMPRESSED CONTEXT:")
    print("=" * 60)
    print(compressed_context)