import chromadb
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),  # can be used with an `url` to load files from a private storage
    dim=384,
    model_file="onnx/model.onnx",  # can be used to load an already supported model with another optimization or quantization, e.g. onnx/model_O4.onnx
)

from typing import Optional

from parser import os




class LectureIndex:
    """
    Индексирует фрагменты текста в векторную базу данных.
    
    Использует multilingual-e5-small для эмбеддингов
    и ChromaDB для хранения и поиска.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        db_path: Optional[str] = None
    ):
        """
        Args:
            model_name: название модели для эмбеддингов
            db_path: путь для сохранения базы на диск.
                     Если None — база живёт только в оперативной памяти.
        """
        local_model_path = os.path.join("models", model_name.replace("/", "_"))

        if os.path.exists(local_model_path):
            print(f"Загрузка модели из локальной папки: {local_model_path}...")
            self.embedder = TextEmbedding(model_name)
        else:
            print(f"Модель не найдена локально. Скачивание {model_name}...")
            self.embedder = TextEmbedding(model_name)
            # Сохраняем, чтобы в следующий раз не скачивать
            self.embedder.save(local_model_path)
            print(f"Модель сохранена в: {local_model_path}")

        print("Модель загружена.")

        if db_path:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        else:
            self.chroma_client = chromadb.Client()

        self.collection = None

    def index_chunks(
        self,
        chunks: list[str],
        collection_name: str = "current_lesson"
    ):
        """
        Векторизует список фрагментов и сохраняет в ChromaDB.
        Если коллекция с таким именем уже существует — пересоздаёт.

        Args:
            chunks: список текстовых фрагментов
            collection_name: имя коллекции в ChromaDB
        """
        # Удаляем старую коллекцию, если есть
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Модель e5 требует префикс "query: " или "passage: "
        # Для индексации документов используется "passage: "
        prefixed = ["passage: " + chunk for chunk in chunks]

        print(f"Векторизация {len(chunks)} фрагментов...")
        embeddings = list(self.embedder.embed(
            prefixed,
            show_progress_bar=True,
            normalize_embeddings=True
        ))

        # Сохраняем в ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

        print(f"Индексация завершена. Фрагментов в базе: {self.collection.count()}")

    def search(self, query: str, n_results: int = 5) -> list[str]:
        """
        Поиск релевантных фрагментов по текстовому запросу.

        Args:
            query: поисковый запрос (тема, вопрос, ключевые слова)
            n_results: сколько фрагментов вернуть

        Returns:
            список найденных фрагментов, отсортированных по релевантности
        """
        if self.collection is None or self.collection.count() == 0:
            print("База пуста. Сначала вызовите index_chunks().")
            return []

        # Для поискового запроса используется префикс "query: "
        query_embedding = list(self.embedder.embed(
            ["query: " + query],
            normalize_embeddings=True
        ))

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count())
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        # Выводим расстояния для отладки
        for i, (doc, dist) in enumerate(zip(documents, distances)):
            preview = doc[:80].replace("\n", " ")
            print(f"  [{i+1}] (расстояние: {dist:.4f}) {preview}...")

        return documents





if __name__ == "__main__":
    from parser import extract_text
    from chunk import chunk_text

    path = 'test_text/1 engl.docx'
    query = 'Bluetooth'

    # Шаг 1: парсинг
    text = extract_text(path)
    print(f"Извлечено {len(text)} символов")

    # Шаг 2: чанкинг
    chunks = chunk_text(text)
    print(f"Разбито на {len(chunks)} фрагментов")

    # Шаг 3: индексация
    index = LectureIndex()
    index.index_chunks(chunks)

    # Проверяем поиск
    print(f"\nПоиск по запросу: «{query}»\n")
    results = index.search(query, n_results=3)

    for i, doc in enumerate(results):
        print(f"\n{'='*50}")
        print(f"Результат {i+1}:")
        print(f"{'='*50}")
        print(doc)