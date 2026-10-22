from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # install langchain-classic
from dotenv import load_dotenv

load_dotenv()

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Initialize Google Generative AI Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Create FAISS vector store using Google embeddings
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

# Create simple retrievers
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# MultiQueryRetriever uses an LLM to generate alternative queries for better recall
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # who will work as retriever, already defined above
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash") #internally which llm to use to create diff query
)

# Query
query = "How to improve energy levels and maintain balance?"

print(f"Executing query: '{query}'\n")

# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

print("=== Standard Similarity Search Results ===")
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*" * 100)

print("=== MultiQueryRetriever Results ===")
for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

#Model Switch: 
# GoogleGenerativeAIEmbeddings seamlessly hooks into FAISS identically to OpenAI embeddings because both inherit from LangChain's core Embeddings base class.

# Hybrid Providers: 
# This script uses Google for generating vector embeddings and OpenAI (ChatOpenAI) to power the MultiQueryRetriever LLM prompt generation. If you prefer keeping everything inside Google's ecosystem, you can also substitute the LLM with ChatGoogleGenerativeAI(model="gemini-1.5-pro") or gemini-2.0-flash.