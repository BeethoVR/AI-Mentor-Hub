import json
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
docs = [
    Document(page_content="p1", metadata={"source": "a", "page": 1}),
    Document(page_content="p2", metadata={"source": "a", "page": 2}),
    Document(page_content="p3", metadata={"source": "a", "page": 3}),
]
store = DocArrayInMemorySearch.from_documents(docs, embeddings)

paginas_objetivo = {("a", 1), ("a", 2)}
docs_expandidos = []
for doc in store.doc_index._docs:
    if (doc.metadata.get("source"), doc.metadata.get("page")) in paginas_objetivo:
        docs_expandidos.append(doc)
print(len(docs_expandidos))
