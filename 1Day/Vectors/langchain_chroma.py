

# https://colab.research.google.com/drive/1jEtDSVVpLv3kicZC2ADmPAJ8J_V1wFWg#scrollTo=X5k__vrf2uTL
# https://chatgpt.com/c/6ad79b19-9e84-83ec-b755-5b978a197e8c

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma

# # building doc object tell about schema in which you want data to be stored.

# from langchain_core.documents import Document

# # Create LangChain documents for IPL players

# doc1 = Document(
#         page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
#         metadata={"team": "Royal Challengers Bangalore"}
#     )
# doc2 = Document(
#         page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
#         metadata={"team": "Mumbai Indians"}
#     )
# doc3 = Document(
#         page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
#         metadata={"team": "Chennai Super Kings"}
#     )
# doc4 = Document(
#         page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
#         metadata={"team": "Mumbai Indians"}
#     )
# doc5 = Document(
#         page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
#         metadata={"team": "Chennai Super Kings"}
#     )


# docs = [doc1, doc2, doc3, doc4, doc5] #list to store in collection

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-2-preview"
# )

# vector_store = Chroma( # give data(docs) you want to store, and where is provider to convert into embeddings
#     embedding_function = embeddings,
#     persist_directory='my_chroma_db',# where to store in root folder , location
#     collection_name='sample' # collection name: sample
# )

# #chroma.sqlite3, vector data stored in this format sqlite3
# # download and can run at sql server

# vector_store.add_documents(docs) 

# # view documents -> tells what you want to see in your db
# vector_store.get(include=['embeddings','documents', 'metadatas'])

# # search documents
# vector_store.similarity_search(
#     query='Who among these are a bowler?',
#     k=2 # no.. of similar object you want to get.
# )


# # search with similarity score
# vector_store.similarity_search_with_score(
#     query='Who among these are a bowler?',
#     k=2
# )

# # meta-data filtering
# vector_store.similarity_search_with_score(
#     query="",
#     filter={"team": "Chennai Super Kings"}
# ) # low score more near to answer (confident answer)


# # update documents
# updated_doc1 = Document(
#     page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
#     metadata={"team": "Royal Challengers Bangalore"}
# )

# vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# vector_store.get(include=['embeddings','documents', 'metadatas'])

# # delete document , any numner of ids can be provided
# vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])

# # view documents
# vector_store.get(include=['embeddings','documents', 'metadatas'])


#--------------------------------------------------------------------------------


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ============================================================
# 1. CREATE DOCUMENTS
# ============================================================
# Document(
#     page_content="...",  # Main text to be embedded and searched
#     metadata={...}       # Structured information for filtering
# )

doc1 = Document(
    page_content=(
        "Virat Kohli is one of the most successful and consistent "
        "batsmen in IPL history. Known for his aggressive batting "
        "style and fitness, he has led Royal Challengers Bangalore "
        "in multiple seasons."
    ),
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content=(
        "Rohit Sharma is one of the most successful captains in "
        "IPL history, leading Mumbai Indians to five titles. "
        "He is known for his calm demeanor and ability to play "
        "big innings under pressure."
    ),
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content=(
        "MS Dhoni, famously known as Captain Cool, has led "
        "Chennai Super Kings to multiple IPL titles. His finishing "
        "skills, wicketkeeping, and leadership are legendary."
    ),
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content=(
        "Jasprit Bumrah is considered one of the best fast bowlers "
        "in T20 cricket. Playing for Mumbai Indians, he is known "
        "for his yorkers and death-over expertise."
    ),
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content=(
        "Ravindra Jadeja is a dynamic all-rounder who contributes "
        "with both bat and ball. Representing Chennai Super Kings, "
        "his quick fielding and match-winning performances make "
        "him a key player."
    ),
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]


# ============================================================
# 2. INITIALIZE THE EMBEDDING MODEL
# ============================================================
# Embeddings convert text into numerical vectors.
# These vectors help Chroma find semantically similar documents.
#
# Make sure the selected model is supported by your API key
# and the installed langchain-google-genai integration.

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


# ============================================================
# 3. INITIALIZE THE CHROMA VECTOR STORE
# ============================================================
# embedding_function: converts text into embeddings
# persist_directory: folder where the database is stored
# collection_name: logical collection inside Chroma

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./my_chroma_db",
    collection_name="sample"
)


# ============================================================
# 4. INSERT DOCUMENTS WITH STABLE IDS
# ============================================================
# Each document gets a unique ID.
# Explicit IDs make it easy to update or delete specific documents.
#
# Delete these IDs first so that rerunning this demonstration
# replaces these five records instead of retaining stale versions.
# Other records in the collection are not intentionally removed.

doc_ids = [
    "player_kohli",
    "player_rohit",
    "player_dhoni",
    "player_bumrah",
    "player_jadeja"
]

vector_store.delete(ids=doc_ids)

inserted_ids = vector_store.add_documents(
    documents=docs,
    ids=doc_ids
)

print("Inserted document IDs:", inserted_ids)


# ============================================================
# 5. VIEW STORED DOCUMENTS
# ============================================================
# get() retrieves stored records directly, without semantic search.
#
# include accepts fields such as:
# "documents"  -> stored text
# "metadatas"  -> stored metadata
# "embeddings" -> numerical vectors

stored_data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nStored documents:")
print(stored_data)

# To inspect embeddings too, use:
# print(vector_store.get(include=[
#     "documents", "metadatas", "embeddings"
# ]))


# ============================================================
# 6. SEMANTIC SIMILARITY SEARCH
# ============================================================
# query: natural-language question or search text
# k: maximum number of results to return
#
# The query is embedded and compared with stored document vectors.

results = vector_store.similarity_search(
    query="Who among these players is a bowler?",
    k=2
)

print("\nSimilarity search results:")

for doc in results:
    print("Text:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("-" * 50)


# ============================================================
# 7. SIMILARITY SEARCH WITH DISTANCE SCORES
# ============================================================
# Returns tuples of (Document, score).
#
# With Chroma's default squared-L2 distance:
# lower score generally means greater similarity.
# This score is NOT a probability or confidence percentage.

scored_results = vector_store.similarity_search_with_score(
    query="Who among these players is a bowler?",
    k=2
)

print("\nSearch results with distance scores:")

for doc, score in scored_results:
    print("Text:", doc.page_content)
    print("Team:", doc.metadata.get("team"))
    print("Distance score:", score)
    print("-" * 50)


# ============================================================
# 8. METADATA FILTERING
# ============================================================
# filter={"team": "Chennai Super Kings"}
# restricts semantic search to documents whose metadata matches.
#
# The query is still required for similarity search.
# Filtering and semantic similarity search work together.

filtered_results = vector_store.similarity_search_with_score(
    query="Which players contribute with bat and ball?",
    k=5,
    filter={"team": "Chennai Super Kings"}
)

print("\nChennai Super Kings search results:")

for doc, score in filtered_results:
    print("Text:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("Distance score:", score)
    print("-" * 50)


# ============================================================
# 9. RETRIEVE RECORDS USING A METADATA FILTER ONLY
# ============================================================
# get(where=...) retrieves matching stored records directly.
# It does not rank them by semantic similarity.

csk_records = vector_store.get(
    where={"team": "Chennai Super Kings"},
    include=["documents", "metadatas"]
)

print("\nAll Chennai Super Kings records:")
print(csk_records)


# ============================================================
# 10. UPDATE AN EXISTING DOCUMENT
# ============================================================
# update_document(document_id=..., document=...)
# requires the ID of an existing document.
#
# The document ID remains the same; its content and metadata
# are replaced with the updated values.

updated_doc1 = Document(
    page_content=(
        "Virat Kohli, the former captain of Royal Challengers "
        "Bangalore (RCB), is renowned for his aggressive leadership "
        "and consistent batting performances. He has scored "
        "numerous runs in IPL history, including multiple centuries. "
        "His ability to chase targets and anchor innings has made "
        "him one of the most dependable batters in T20 cricket."
    ),
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id="player_kohli",
    document=updated_doc1
)

print("\nUpdated Kohli document:")
print(
    vector_store.get(
        ids=["player_kohli"],
        include=["documents", "metadatas"]
    )
)


# ============================================================
# 11. DELETE DOCUMENTS
# ============================================================
# delete(ids=[...]) deletes the specified document IDs.
# Multiple IDs can be passed in the same list.

vector_store.delete(
    ids=["player_kohli"]
)

print("\nKohli document after deletion:")
print(
    vector_store.get(
        ids=["player_kohli"],
        include=["documents", "metadatas"]
    )
)


# ============================================================
# 12. VERIFY THE FINAL COLLECTION
# ============================================================
# Kohli has been deleted; the other four documents remain.

final_data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nFinal collection:")
print(final_data)

print("\nRemaining document count:", len(final_data["ids"]))
