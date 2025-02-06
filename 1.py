import re
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
import numpy as np
import faiss
from langchain_ollama import  OllamaLLM
reader=PdfReader("doc1.pdf")
all_paragraphs=[]
text=reader.pages

for page in text:
    text_of_page=page.extract_text()
    if text_of_page:
        paragraphs=re.split(r'\.\n',text_of_page)
        non_empty_paragraphs = [para.strip() for para in paragraphs if para.strip()]
        all_paragraphs.extend(non_empty_paragraphs)


embedding_model=OllamaEmbeddings(model="nomic-embed-text")
embeddings=embedding_model.embed_documents(all_paragraphs)
embeddings = np.array(embeddings)
print(all_paragraphs[0])
print(f"Размерность эмбеддингов: {embeddings.shape}")


index = faiss.IndexFlatL2(embeddings.shape[1])
batch_size=100
for i in range(0, len(embeddings), batch_size):
    index.add(embeddings[i:i+batch_size])
query = "Насколько ?"
query_embedding=embedding_model.embed_query(query)
query_embedding = np.array(query_embedding)
print(query_embedding)
k = 10
indexes=index.search(query_embedding.reshape(1,-1),k)[1][0]
for i in indexes:
    print(all_paragraphs[i])
context = [text.replace("\xa0", " ") for text in [all_paragraphs[i] for i in indexes]]
context=[text.replace("\n", " ") for text in context]

input=f"Вот контекст из документа, выбери из него подходящую информацию для вопроса пользователья и отвечай  по контексту. Если в контексте нет информации близкой к вопросу, просто дословно напиши 'Я не знаю'. Контекст:\n\n{context}\n\nВопрос пользователя: {query}"

llm=OllamaLLM(model="llama3.2")
response=llm.invoke(input)
print(response)
#distances, indices = index.search(query_embedding, k)
# print("\n🔍 Запрос:", query)
# for i, idx in enumerate(indices[0]):
#     print(f"Результат {i+1}: {all_paragraphs[idx]} (дистанция: {distances[0][i]:.4f})")
# reader2=PdfReader("doc2.pdf")
# text2=reader2.pages
# for page2 in text2:
#     text_of_page2=page2.extract_text()
#     print(text_of_page2)
#     paragraphs2=re.split(r'\n',text_of_page2)
#     non_empty_paragraphs2=[para.strip() for para in paragraphs2 if para.strip()]
#     all_paragraphs2.extend(non_empty_paragraphs2)
# print(all_paragraphs2[0])