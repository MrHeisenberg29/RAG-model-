from pypdf import PdfReader
import re
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_ollama import OllamaLLM


# 1️⃣ Функция для извлечения абзацев
def extract_paragraphs(pdf_path):
    # Чтение PDF
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    # Разделение текста на абзацы по двойному переносу строки или точке с двумя новыми строками
    paragraphs = re.split(r"(\n\s*\n|\.\s*\n\s*\n)", text)

    # Объединяем части текста, игнорируя пустые строки и ненужные разделители
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

    return paragraphs


# 2️⃣ Загружаем PDF и извлекаем абзацы
pdf_path = "doc1.pdf"
paragraphs = extract_paragraphs(pdf_path)
print(paragraphs)

# 3️⃣ Создаём эмбеддинги (векторное представление текста)
embedding_model = OllamaEmbeddings(model="deepseek-r1")
documents = [Document(page_content=p, metadata={"source": f"Paragraph {i + 1}"}) for i, p in enumerate(paragraphs)]

# Создаём векторное хранилище на основе FAISS
vector_db = FAISS.from_documents(documents, embedding_model)


# 4️⃣ Функция для поиска релевантного контекста
def find_relevant_paragraphs(query, top_k=3):
    docs = vector_db.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

context=find_relevant_paragraphs("Что такое абонентский номер?")
print("Контекст для запроса:\n", context)
# 5️⃣ Функция для запроса к модели Ollama
def query_ollama(query):
    context = find_relevant_paragraphs(query)
    print("Контекст для запроса:\n", context)

    full_prompt = f"Вот контекст из документа, отвечай в точности по этому контексту:\n\n{context}\n\nОтветь на вопрос в точности как в контексте: {query}"

    llm = OllamaLLM(model="deepseek-r1")  # Убедитесь, что модель корректна
    response = llm.invoke(full_prompt)  # Используем метод invoke()

    return response


# 6️⃣ Пример запроса
user_query = "Что такое Абонентский номер?"
response = query_ollama(user_query)
print("\n💡 Ответ модели:", response)
