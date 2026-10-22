# pip install langchain langchain-openai langchain-community faiss-cpu

import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Optional: Set your OpenAI API key if not already in your environment variables
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def main():
    # Step 1: Define sample documents
    docs = [
        Document(page_content="LangChain makes it easy to work with LLMs."),
        Document(page_content="LangChain is used to build LLM based applications."),
        Document(page_content="Chroma is used to store and search document embeddings."),
        Document(page_content="Embeddings are vector representations of text."),
        Document(page_content="MMR helps you get diverse results when doing similarity search."),
        Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
    ]

    # Step 2: Initialize OpenAI embeddings model
    embedding_model =  GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview"
    )

    # Step 3: Create the FAISS vector store from documents
    # FAISS (Facebook AI Similarity Search) is an efficient library for similarity search and clustering of dense vectors.
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    # Step 4: Configure the Retriever with MMR (Maximum Marginal Relevance)
    # - search_type="mmr": Avoids redundant results by optimizing for both relevance AND diversity.
    # - k: Number of final documents to return.
    # - lambda_mult: Controls the balance. 
    #   * 1.0 = Pure similarity (standard search, might return identical/redundant chunks).
    #   * 0.0 = Maximum diversity (results might stray far from the query topic).
    #   * 0.5 = Balanced trade-off (recommended default).
    retriever = vectorstore.as_retriever(
        search_type="mmr",                    
        search_kwargs={"k": 3, "lambda_mult": 0.5}  
    )

    # Define query
    query = "What is langchain?"
    print(f"Executing MMR query: '{query}'...\n")

    # Step 5: Invoke the retriever
    results = retriever.invoke(query)

    # Step 6: Display output
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content)

if __name__ == "__main__":
    main()


#Key Takeaways on MMR (Maximum Marginal Relevance)Why use MMR? 
# Standard similarity search looks only at how close a document's vector is to the query vector. 
# If your corpus has multiple very similar paragraphs, standard search might return 3 nearly identical sentences. 
# MMR penalizes documents that are too similar to ones already selected, ensuring a diverse set of contexts for your LLM.\
# The Role of lambda_mult:1.0: Pure relevance (standard top-$k$ vector search).0.5: A 50/50 balance between matching the query closely and bringing in fresh, complementary information.