import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# pip install langchain langchain-openai langchain-community chromadb




# Optional: Set your OpenAI API key programmatically if not set in your environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

def main():
    # Step 1: Define your source documents
    documents = [
        Document(page_content="LangChain helps developers build LLM applications easily."),
        Document(page_content="Chroma is a vector database optimized for LLM-based search."),
        Document(page_content="Embeddings convert text into high-dimensional vectors."),
        Document(page_content="OpenAI provides powerful embedding models."),
    ]

    # Step 2: Initialize the OpenAI embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview"
        )

    # Step 3: Create a Chroma vector store in memory
    # This embeds your documents and stores them locally/in-memory
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name="my_collection"
    )

    # Define your search query
    query = "What is Chroma used for?"

    # ==========================================
    # Approach 1: Using LangChain's Retriever Interface
    # ==========================================
    # .as_retriever() wraps the vectorstore into LangChain's standardized Runnable protocol.
    # This is ideal when plugging into larger RAG chains (like conversational retrieval chains).
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retriever_results = retriever.invoke(query)

    print("=== Results using Retriever (.invoke) ===")
    for i, doc in enumerate(retriever_results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)

    # ==========================================
    # Approach 2: Using Direct VectorStore Search
    # ==========================================
    # .similarity_search() queries the vector database directly without the Retriever abstraction layer.
    # Useful for quick lookups or direct database querying.
    search_results = vectorstore.similarity_search(query, k=2)

    print("\n=== Results using Direct similarity_search ===")
    for i, doc in enumerate(search_results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)

if __name__ == "__main__":
    main()


### Key Takeaway: Retriever vs. VectorStore Search

# * **`vectorstore.as_retriever().invoke(query)`**:
#  Converts the database into a standardized LangChain component (`Runnable`). This is best when you want to pass the retriever into complex pipelines like `create_retrieval_chain`.
# * **`vectorstore.similarity_search(query, k=2)`**:
#  Directly queries the underlying Chroma instance. It bypasses LangChain's retriever abstraction layer, making it great for lightweight standalone script testing or quick debugging.
