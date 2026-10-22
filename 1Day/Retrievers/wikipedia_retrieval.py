
# pip install langchain-community wikipedia

import wikipedia
from langchain_community.retrievers import WikipediaRetriever

# 1. FIX: Set a custom, descriptive User-Agent to comply with Wikimedia's policy
# Format usually follows: AppName/Version (Contact Information/Email)
wikipedia.set_user_agent("LangChainLearningApp/1.0 (ujjwaldikshit1@gmail.com)")
    
def main():
    # 1. Initialize the Wikipedia Retriever
    # - top_k_results: Limits the maximum number of Wikipedia pages returned.
    # - lang: Sets the language edition of Wikipedia (e.g., "en" for English).
    retriever = WikipediaRetriever(top_k_results=2, lang="en")

    # 2. Define your search query
    # Note: WikipediaRetriever performs keyword and title matching via the Wikipedia API,
    # rather than semantic vector search. Specific or multi-concept queries will fetch 
    # the closest matching articles (e.g., India–Pakistan relations, Sino-Pakistani relations).
    query = "the geopolitical history of india and pakistan from the perspective of a chinese"

    print(f"Querying Wikipedia for: '{query}'...\n")

    # 3. Retrieve relevant documents using LangChain's Runnable protocol (.invoke())
    # .invoke() sends the query to the underlying Wikipedia API and wraps the results 
    # into a list of LangChain Document objects.
    docs = retriever.invoke(query)

    # 4. Process and display the retrieved content and metadata
    for i, doc in enumerate(docs):
        print(f"--- Result {i + 1} ---")
        
        # Display available metadata (such as the source URL or article title)
        if hasattr(doc, "metadata") and doc.metadata:
            print(f"Title / Source: {doc.metadata.get('title', 'N/A')}")
            print(f"URL: {doc.metadata.get('source', 'N/A')}")
        
        # Print a truncated preview of the page content
        # doc.page_content contains the raw text extracted from the Wikipedia page
        preview_text = doc.page_content[:400].strip()
        print(f"Content Preview:\n{preview_text}...\n")

if __name__ == "__main__":
    main()


# ### Key Theoretical Concepts & Learning Points

# * **The LangChain Runnable Protocol (`.invoke`)**:
# Modern LangChain components (including retrievers, chains, and models) inherit from the `Runnable` interface. The `.invoke()` method is the standard standard way to execute a component with a single input. Older versions of LangChain used `.get_relevant_documents(query)`, which has now been standardized to `.invoke(query)`.
# * **How `WikipediaRetriever` Works**:
# Unlike a Vector Store Retriever (which uses semantic embeddings and cosine similarity), the `WikipediaRetriever` relies on the **Wikipedia API's built-in text search**. It matches keywords and article titles, making it great for fetching real-time, factual encyclopedic summaries without needing an embedding model or vector database.
# * **Structure of a LangChain `Document**`:
# The `docs` returned are a list of `Document` objects. Each document contains two main attributes:
# 1. `page_content` ($str$): The actual textual data scraped from the article.
# 2. `metadata` ($dict$): Dictionary containing helpful contextual data like the article's title, summary snippet, and source link.
#------------------------------------------------------------------------------------------------------

# this is more like document_loader
# no, based on search query=> giving relevant document => have some form of intelligent.