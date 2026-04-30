from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import os
import time


@dataclass
class CompressionConfig:
    # Сколько чанков сжимать максимум (если retrieval вернул слишком много)
    max_chunks: int = 30

    # Map step: бюджет на 1 чанк (в токенах вывода)
    per_chunk_max_tokens: int = 180

    # Reduce step: бюджет финального сжатого контекста (в токенах вывода)
    final_max_tokens: int = 900

    # Генерация
    temperature: float = 0.1
    top_p: float = 0.9

    # Ограничение на общий размер промпта под n_ctx (чтобы не переполнять контекстное окно)
    # Это не идеально, но практично.
    max_prompt_tokens: int = 3200


# ----------------------------
# Core compressor
# ----------------------------

class ContextCompressor:
    def __init__(self, compressor_llm):
        self.gen = compressor_llm

    def count_tokens(self, text: str) -> int:
        """
        Точное число токенов через llama.cpp tokenizer (если доступно).
        Если не доступно — грубая оценка.
        """
        try:
            return len(self.gen.llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            return max(1, len(text) // 4)

    def truncate_to_token_budget(self, text: str, max_tokens: int) -> str:
        """
        Грубая обрезка текста, если он не влезает по токенам.
        (Обрезаем по символам, т.к. "детокенизатора" может не быть.)
        """
        if self.count_tokens(text) <= max_tokens:
            return text

        # прибл. 1 токен ~ 4 символа
        target_chars = max(200, max_tokens * 4)
        return text[:target_chars] + "\n\n[...ОБРЕЗАНО ПО БЮДЖЕТУ...]"

    # ---------- prompt templates (with <|im_start|>/<|im_end|> placeholders) ----------

    def build_map_prompt(self, chunk: str, query: str) -> str:
        focus = f"ТЕМА/ФОКУС: {query}\n" if query.strip() else ""
        return f"""<|im_start|> system
Ты — инструмент для сжатия учебного текста.
Правила:
- Пиши ТОЛЬКО то, что явно содержится в исходном фрагменте.
- Ничего не добавляй от себя.
- Если факт не подтверждается фрагментом — не включай его.
- Предпочитай определения, причины/следствия, перечисления, формулы/обозначения, ключевые шаги.
<|im_end|>
<|im_start|> user
Сожми фрагмент в список опорных фактов для последующей генерации тестовых вопросов.

{focus}
ФРАГМЕНТ:
{chunk}

ФОРМАТ ОТВЕТА (строго):
- 5–12 пунктов, каждый пункт — краткий факт/определение/правило (1–2 предложения)
- Не повторяй одинаковые мысли
- Не используй общие фразы, только содержание
<|im_end|>
<|im_start|> assistant
"""

    def build_reduce_prompt(self, compressed_chunks: List[str], query: str) -> str:
        focus = f"ТЕМА/ФОКУС: {query}\n" if query.strip() else ""
        joined = "\n\n".join([f"[СЖАТИЕ ЧАНКА #{i+1}]\n{c}" for i, c in enumerate(compressed_chunks)])

        return f"""<|im_start|> system
Ты — инструмент для финального сжатия и нормализации опорных фактов.
Правила:
- Используй только то, что присутствует в предоставленных пунктах.
- Удали повторы и переформулируй дубликаты в один канонический пункт.
- Сохрани максимальную информативность при минимальном объёме.
<|im_end|>
<|im_start|> user
Объедини и дополнительно сожми опорные факты из нескольких чанков в единый компактный конспект для генерации тестовых вопросов.

{focus}
ВХОДНЫЕ ОПОРНЫЕ ФАКТЫ:
{joined}

ТРЕБОВАНИЯ К ВЫХОДУ:
- 12–25 пунктов (если материала мало — меньше)
- Пункты сгруппируй по смысловым блокам с короткими заголовками вида: "Термины:", "Процесс:", "Свойства:", "Примеры:"
- Никаких рассуждений о том, что ты сделал — только итог
<|im_end|>
<|im_start|> assistant
"""

    # ---------- main API ----------

    def compress_chunks(
        self,
        chunks: List[str],
        query: str = "",
        cfg: Optional[CompressionConfig] = None,
    ) -> Tuple[str, Dict]:
        """
        Returns:
          (compressed_context, debug_info)
        """
        cfg = cfg or CompressionConfig()

        t0 = time.time()

        chunks = [c.strip() for c in chunks if c and c.strip()]
        if not chunks:
            return "", {"n_chunks_in": 0, "n_chunks_used": 0}

        chunks = chunks[: cfg.max_chunks]

        # ---- MAP ----
        compressed_parts: List[str] = []
        map_times: List[float] = []

        for i, chunk in enumerate(chunks):
            map_prompt = self.build_map_prompt(chunk=chunk, query=query)

            # защитимся от переполнения контекста
            map_prompt = self.truncate_to_token_budget(map_prompt, cfg.max_prompt_tokens)

            t1 = time.time()
            out = self.gen.generate_single(
                prompt=map_prompt,
                temperature=cfg.temperature,
            )
            map_times.append(time.time() - t1)

            out = out.strip()
            if out:
                compressed_parts.append(out)

        if not compressed_parts:
            return "", {
                "n_chunks_in": len(chunks),
                "n_chunks_used": len(chunks),
                "map_ok": 0,
                "reduce_ok": 0,
            }

        # ---- REDUCE ----
        reduce_prompt = self.build_reduce_prompt(compressed_chunks=compressed_parts, query=query)
        reduce_prompt = self.truncate_to_token_budget(reduce_prompt, cfg.max_prompt_tokens)

        t2 = time.time()
        final = self.gen.generate_single(
            prompt=reduce_prompt,
            temperature=cfg.temperature,
        )
        reduce_time = time.time() - t2

        final = final.strip()

        debug = {
            "n_chunks_in": len(chunks),
            "n_chunks_used": len(chunks),
            "map_ok": len(compressed_parts),
            "map_time_avg_sec": sum(map_times) / max(1, len(map_times)),
            "reduce_time_sec": reduce_time,
            "total_time_sec": time.time() - t0,
        }
        return final, debug



if __name__ == "__main__":
    # Пример показывает только идею: retrieval -> compress -> печать.
    # Подразумевается, что у вас уже есть retrieval, возвращающий список чанков.
    #
    # Для реального запуска:
    # 1) установите env CHAT_START_TOKEN и CHAT_END_TOKEN в свои настоящие токены
    # 2) используйте более слабую модель для compressor (например, 0.5B–1.5B),
    #    а основную — для генерации вопросов.

    from parser import extract_text
    from chunk import chunk_text
    from embedder import LectureIndex
    from retrieval.dpp_retrieval import DPPContextRetriever
    from generator import LLMGenerator

    path = "test_text/1 engl.docx"
    topic = "Bluetooth"

    # Индексация
    text = extract_text(path)
    chunks = chunk_text(text)
    index = LectureIndex()
    index.index_chunks(chunks)

    # Retrieval (например, DPP)
    retriever = DPPContextRetriever(index)
    context, selected = retriever.get_context(topic=topic, top_m=15, n_select=6)

    # Компрессор (можно выбрать более слабую модель)
    compressor_llm = LLMGenerator(model_path="D:/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    compressor = ContextCompressor(compressor_llm)

    # Сжимаем именно выбранные чанки (а не весь склеенный контекст)
    compressed, info = compressor.compress_chunks(selected, query=topic)

    print("DEBUG:", info)
    print("\n" + "=" * 60)
    print("COMPRESSED CONTEXT:")
    print("=" * 60)
    print(compressed)