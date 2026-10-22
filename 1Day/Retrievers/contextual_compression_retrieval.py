import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Optional: Set your OpenAI API key if not already configured in your environment
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def main():
    # Step 1: Define documents containing both relevant facts and unrelated noise/distractors
    docs = [
        Document(page_content=(
            "The Grand Canyon is one of the most visited natural wonders in the world. "
            "Photosynthesis is the process by which green plants convert sunlight into energy. "
            "Millions of tourists travel to see it every year. The rocks date back millions of years."
        ), metadata={"source": "Doc1"}),

        Document(page_content=(
            "In medieval Europe, castles were built primarily for defense. "
            "The chlorophyll in plant cells captures sunlight during photosynthesis. "
            "Knights wore armor made of metal. Siege weapons were often used to breach castle walls."
        ), metadata={"source": "Doc2"}),

        Document(page_content=(
            "Basketball was invented by Dr. James Naismith in the late 19th century. "
            "It was originally played with a soccer ball and peach baskets. NBA is now a global league."
        ), metadata={"source": "Doc3"}),

        Document(page_content=(
            "The history of cinema began in the late 1800s. Silent films were the earliest form. "
            "Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells. "
            "Modern filmmaking involves complex CGI and sound design."
        ), metadata={"source": "Doc4"})
    ]

    # Step 2: Initialize embeddings and create a FAISS vector store
    embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Step 3: Create a standard base retriever to cast a wide net (fetch top candidate documents)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Step 4: Set up the LLM compressor using LLMChainExtractor
    # LLMChainExtractor passes each retrieved document through an LLM to strip away 
    # irrelevant sentences, leaving behind only the exact information pertinent to the query.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature = 0)
    compressor = LLMChainExtractor.from_llm(llm)

    # Step 5: Combine the base retriever and compressor into a ContextualCompressionRetriever
    # This pipeline first fetches raw chunks via vector similarity, then filters/compresses them.
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor # basically a llm
    )

    # Step 6: Query the compression retriever
    query = "What is photosynthesis?"
    print(f"Executing compressed retrieval for query: '{query}'\n")
    
    compressed_results = compression_retriever.invoke(query)

    # Step 7: Display the cleaned, extracted snippets
    for i, doc in enumerate(compressed_results):
        print(f"--- Compressed Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content:\n{doc.page_content}\n")

if __name__ == "__main__":
    main()

# Key Concepts & Learning Points

# The Problem with Standard Retrieval: Standard vector search returns entire document chunks. If a chunk contains 90% unrelated text and 10% relevant information, that "noise" is passed to your LLM, which can dilute response accuracy and increase token costs.

# How ContextualCompressionRetriever Works:

# Phase 1 (Fetch): Uses a base retriever (like FAISS) to gather a broad set of candidate documents.

# Phase 2 (Compress): Passes those documents through a document compressor (such as LLMChainExtractor) to prune out sentences that do not directly address the query.

# Benefits: Keeps your LLM context window clean, focused, and precise, resulting in significantly higher quality RAG generations.