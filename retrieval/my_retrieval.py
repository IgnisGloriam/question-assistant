try:
    import import_module
except:
    pass

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

from embedder import LectureIndex


# ----------------------------
# Sentence splitting helpers
# ----------------------------

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
        # делим по (.!? + пробел/конец)
        raw = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        return [s.strip() for s in raw if s.strip()]


# ----------------------------
# Union-Find for grouping duplicates
# ----------------------------

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


# ----------------------------
# Data container
# ----------------------------

@dataclass
class SentenceRec:
    paragraph_id: int
    sent_id: int          # глобальный порядок предложения в контексте
    text: str


# ----------------------------
# Main retriever
# ----------------------------

class MyContextRetriever:
    """
    Делает обычный retrieval (Chroma top-k),
    затем чистит контекст: удаляет семантически похожие предложения,
    сохраняя структуру (абзацы и порядок).
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
        n_results: int = 5,
        max_context_chars: int = 3000,
        # параметры дедупликации:
        dedup: bool = True,
        sim_threshold: float = 0.92,
        n_projections: int = 4,
        neighbor_window: int = 2,
        min_sentence_chars: int = 25,
    ) -> str:
        # 1) Retrieval (как раньше)
        if topic.strip():
            fragments = self._search_by_topic(topic, n_results)
        else:
            fragments = self._search_broad(n_results)

        # 2) Сборка первичного контекста
        # Лайфхак: можно собрать чуть больше, а потом дедуплицировать и ужать
        raw_context = self._build_context(fragments, max_context_chars=max_context_chars)

        print("\n" + "=" * 60)
        print("RAW:")
        print("=" * 60)
        print(raw_context)

        if not dedup:
            return raw_context

        # 3) Дедупликация предложений внутри контекста
        cleaned_context, info = self._semantic_dedup_sentences(
            text=raw_context,
            query=topic.strip(),
            sim_threshold=sim_threshold,
            n_projections=n_projections,
            neighbor_window=neighbor_window,
            min_sentence_chars=min_sentence_chars,
            max_output_chars=max_context_chars,
        )

        print(
            f"Дедупликация: было {info['n_sentences']}, стало {info['kept_sentences']} "
            f"(удалено {info['removed_sentences']})"
        )

        return cleaned_context

    # ---- Retrieval helpers ----

    def _search_by_topic(self, topic: str, n_results: int) -> List[str]:
        print(f"Поиск по теме: «{topic}»")
        return self.index.search(topic, n_results=n_results)

    def _search_broad(self, n_results_per_query: int) -> List[str]:
        print("Тема не указана, ищем по общим запросам...")

        all_fragments = []
        seen = set()

        for query in self.DEFAULT_QUERIES:
            results = self.index.search(query, n_results=n_results_per_query)
            for fragment in results:
                if fragment not in seen:
                    seen.add(fragment)
                    all_fragments.append(fragment)

        return all_fragments

    def _build_context(self, fragments: List[str], max_context_chars: int) -> str:
        parts = []
        total = 0

        for fragment in fragments:
            if total + len(fragment) > max_context_chars:
                remaining = max_context_chars - total
                if remaining > 100:
                    parts.append(fragment[:remaining] + "...")
                elif remaining > 0:
                    parts.append(fragment[:remaining])
                break

            parts.append(fragment)
            total += len(fragment)

        context = "\n\n".join(parts)
        print(f"Контекст собран: {len(context)} символов из {len(parts)} фрагментов")
        return context

    # ---- Semantic sentence dedup ----

    def _semantic_dedup_sentences(
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
        Приближённая дедупликация:
          - предложения -> эмбеддинги
          - несколько случайных проекций + сортировка
          - сравниваем только соседей в сортировке (neighbor_window)
          - union-find группы дублей
          - представитель группы: max relevance к query (если задан), иначе центроидный
          - восстановление абзацев и исходного порядка
        """

        paragraphs = split_into_paragraphs(text)

        # собрать предложения
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

        # эмбеддинги предложений (fastembed внутри index)
        X = self.index.embed_texts(
            ["passage: " + t for t in sent_texts],
            normalize=True
        ).astype(np.float32)

        # eligible: короткие предложения не трогаем
        eligible = np.array([len(t) >= min_sentence_chars for t in sent_texts], dtype=bool)

        uf = UnionFind(len(sent_texts))
        rng = np.random.default_rng(seed)

        d = X.shape[1]

        # случайные проекции + локальные сравнения
        for _ in range(n_projections):
            r = rng.normal(size=(d,)).astype(np.float32)
            r /= (np.linalg.norm(r) + 1e-12)

            proj = X @ r
            order = np.argsort(proj)

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

                    sim = float(X[i] @ X[j])  # cosine (X normalized)
                    if sim >= sim_threshold:
                        uf.union(i, j)

        # группы
        groups: Dict[int, List[int]] = {}
        for i in range(len(sent_texts)):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)

        # релевантность к query (если есть)
        sim_to_query = None
        if query.strip():
            q = self.index.embed_texts(["query: " + query], normalize=True)[0].astype(np.float32)
            sim_to_query = (X @ q)

        keep_idx = set()

        for root, idxs in groups.items():
            if len(idxs) == 1:
                keep_idx.add(idxs[0])
                continue

            if sim_to_query is not None:
                best = max(idxs, key=lambda i: float(sim_to_query[i]))
            else:
                mean_vec = X[idxs].mean(axis=0)
                mean_vec /= (np.linalg.norm(mean_vec) + 1e-12)
                best = max(idxs, key=lambda i: float(X[i] @ mean_vec))

            keep_idx.add(best)

        # восстановить порядок и абзацы
        kept = [s for i, s in enumerate(sents) if i in keep_idx]
        kept.sort(key=lambda s: s.sent_id)

        by_par: Dict[int, List[str]] = {pid: [] for pid in range(len(paragraphs))}
        for s in kept:
            by_par[s.paragraph_id].append(s.text)

        out_pars = []
        for pid in range(len(paragraphs)):
            if by_par[pid]:
                out_pars.append(" ".join(by_par[pid]).strip())

        out_text = "\n\n".join(out_pars).strip()

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

    retriever = MyContextRetriever(index)

    context = retriever.get_context(
        topic=topic,
        n_results=5,
        # max_context_chars=3000,
        dedup=True,
        sim_threshold=0.92,
        n_projections=4,
        neighbor_window=2,
        min_sentence_chars=25,
    )

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ КОНТЕКСТ ДЛЯ LLM:")
    print("=" * 60)
    print(context)