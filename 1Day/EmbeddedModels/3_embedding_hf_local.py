from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
text = "Delhi is the captital of india"

vector = embedding.embed_query(text)
print(str(vector))

# python EmbeddedModels/3_embedding_hf_local.py