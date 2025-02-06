import re
import pandas as pd
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
import numpy as np
import faiss
from langchain_ollama import  OllamaLLM


def pdf_to_chunks(pdf_path):
    reader=PdfReader(pdf_path)
    text=reader.pages
    all_chunks=[]
    for page in text:
        text_of_page=page.extract_text()
        if text_of_page:
            if not re.search(r'[.!?]', text_of_page):
                all_chunks.append(text_of_page)
            sentences = re.split(r'(?<=[.!?])\s+', text_of_page)
            chunks = []
            current_chunk = ""
            for i in sentences:
                if len(current_chunk) + len(i) > 500:
                    chunks.append(current_chunk.strip())
                    current_chunk = i
                else:
                    current_chunk += " " + i
            all_chunks.extend(chunks)
    return  all_chunks


def csv_to_chunks(csv_path):
    df = pd.read_csv(csv_path)
    prefixes = ["Услуга", "Условие", "Тариф"]

    all_chunks = df.apply(lambda row: " | ".join(
        [f"{prefixes[i]}: {str(row[col])}" for i, col in enumerate(df.columns) if pd.notna(row[col])]
    ), axis=1).tolist()
    return all_chunks

chunks1=pdf_to_chunks("doc1.pdf")
chunks2=pdf_to_chunks("doc2.pdf")
chunks3=csv_to_chunks("cards.csv")
for i in chunks3:
    print(i)
embedding_model=OllamaEmbeddings(model="mxbai-embed-large")
embeddings=embedding_model.embed_documents(chunks1+chunks2+chunks3)
embeddings = np.array(embeddings)
index=faiss.IndexFlatL2(embeddings.shape[1])
for i in range(0,len(embeddings),100):
    index.add(embeddings[i:i+100])
query = "Кто такой А.С. Пушкин?"
query_embedding=embedding_model.embed_query(query)
query_embedding = np.array(query_embedding)
k=10
indexes=index.search(query_embedding.reshape(1,-1),k)[1][0]
context=[]
for i in indexes:
    if(i<len(chunks1)):
        context.append(chunks1[i])
    elif(i<len(chunks1)+len(chunks2)):
        context.append(chunks2[i-len(chunks1)])
    elif(i<len(chunks1)+len(chunks2)+len(chunks3)):
        context.append(chunks3[i - len(chunks1)-len(chunks3)])
context=[text.replace("\n", " ") for text in context]
context=[text.replace("\xa0", " ") for text in context]
print(len(context))
print(context)

input=f"Вот контекст из документа, выбери из него подходящую информацию для вопроса пользователя и отвечай по контексту. Если в контексте нет информации близкой к вопросу, просто дословно напиши 'Я не знаю'. И не упоминай в тексте ответа о том что контекст был. Контекст:\n\n{context}\n\nВопрос пользователя: {query}"

llm=OllamaLLM(model="llama3.2")
response=llm.invoke(input)
print(response)
# for page in text:
#     text_of_page=page.extract_text()
#     if text_of_page:
#         if not re.search(r'[.!?]', text_of_page):
#             all_paragraphs.append(text_of_page)
#         sentences = re.split(r'(?<=[.!?])\s+', text_of_page)
#         chunks=[]
#         current_chunk=""
#         for i in sentences:
#             if len(current_chunk)+len(i)>500:
#                 chunks.append(current_chunk.strip())
#                 current_chunk=i
#             else:
#                 current_chunk+=" "+i
#         all_paragraphs.extend(chunks)
#
# print(len(all_paragraphs))
# embedding_model=OllamaEmbeddings(model="nomic-embed-text")
# embeddings=embedding_model.embed_documents(all_paragraphs)
# embeddings = np.array(embeddings)
# print(all_paragraphs[0])
# print(f"Размерность эмбеддингов: {embeddings.shape}")
#
#
# index = faiss.IndexFlatL2(embeddings.shape[1])
# batch_size=100
# for i in range(0, len(embeddings), batch_size):
#     index.add(embeddings[i:i+batch_size])
# query = "Номер редакции условий комплексного банковского обслуживания?"
# query_embedding=embedding_model.embed_query(query)
# query_embedding = np.array(query_embedding)
# print(query_embedding)
# k = 10
# indexes=index.search(query_embedding.reshape(1,-1),k)[1][0]
# for i in indexes:
#     print(all_paragraphs[i])
# context = [text.replace("\xa0", " ") for text in [all_paragraphs[i] for i in indexes]]
# context=[text.replace("\n", " ") for text in context]
#
# input=f"Вот контекст из документа, выбери из него подходящую информацию для вопроса пользователья и отвечай  по контексту. Если в контексте нет информации близкой к вопросу, просто дословно напиши 'Я не знаю'. Контекст:\n\n{context}\n\nВопрос пользователя: {query}"
#
# llm=OllamaLLM(model="llama3.2")
# response=llm.invoke(input)
# print(response)
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