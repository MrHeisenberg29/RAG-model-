import re
import pandas as pd
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# --- Улучшенное разбиение текста (Sliding Window) ---
def pdf_to_chunks(pdf_path, chunk_size=500, overlap=100):
    reader = PdfReader(pdf_path)
    all_chunks = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, current_chunk = [], []
            current_length = 0

            for sent in sentences:
                if current_length + len(sent) > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = current_chunk[-overlap // 10:]  # Перекрытие
                    current_length = sum(map(len, current_chunk))
                current_chunk.append(sent)
                current_length += len(sent)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Добавляем метадату с номером страницы
            all_chunks.extend([f"[Стр. {page_num + 1}] {chunk}" for chunk in chunks])

    return all_chunks


def csv_to_chunks(csv_path):
    df = pd.read_csv(csv_path)
    prefixes = ["Услуга", "Условие", "Тариф"]

    all_chunks = df.apply(lambda row: " | ".join(
        [f"{prefixes[i]}: {str(row[col])}" for i, col in enumerate(df.columns) if pd.notna(row[col])]), axis=1).tolist()

    return all_chunks


# --- Читаем файлы ---
chunks1 = pdf_to_chunks("doc1.pdf")
chunks2 = pdf_to_chunks("doc2.pdf")
chunks3 = csv_to_chunks("cards.csv")

all_chunks = chunks1 + chunks2 + chunks3

# --- Улучшенные эмбеддинги и FAISS ---
embedding_model = OllamaEmbeddings(model="nomic-embed-text")  # Лучшая модель
embeddings = embedding_model.embed_documents(all_chunks)
embeddings = np.array(embeddings)

# Нормализация эмбеддингов (важно для косинусного поиска)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# Создаем FAISS индекс (Inner Product ~ косинусное сходство)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# --- Улучшенный поиск ---
query = "Что таккое банк?"

# Обогащение запроса синонимами (LLM query expansion)
llm = OllamaLLM(model="llama3.2")
query_expansion = llm.invoke(f"Перефразируй этот запрос несколькими способами: {query}")
queries = [query] + query_expansion.split("\n")

# Получаем средний эмбеддинг всех вариантов запроса
query_embeddings = np.array([embedding_model.embed_query(q) for q in queries])
query_embedding = np.mean(query_embeddings, axis=0)

# Нормализация запроса
query_embedding /= np.linalg.norm(query_embedding)

# Поиск в FAISS
D, I = index.search(query_embedding.reshape(1, -1), 30)  # Увеличили k до 20
threshold = 0.3  # Порог схожести, можно настроить
filtered_indexes = [i for i, d in zip(I[0], D[0]) if d > threshold]

# Собираем найденные чанки
context = [all_chunks[i] for i in filtered_indexes]
context = [text.replace("\n", " ").replace("\xa0", " ") for text in context]
print(context)
# --- Генерация ответа ---
input_prompt = f"""Вот контекст из документа, выбери из него подходящую информацию для вопроса пользователя 
и отвечай только на основе контекста. Если в контексте нет ответа, напиши 'Я не знаю'. 
Контекст:\n\n{context}\n\nВопрос: {query}"""

response = llm.invoke(input_prompt)

print(response)
