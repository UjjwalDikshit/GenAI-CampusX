#5

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file
# This is where GOOGLE_API_KEY is usually stored.
load_dotenv()

# Create an embedding model.
# Embeddings convert text into numerical vectors that capture semantic meaning.
# SemanticChunker uses these vectors to determine where the topic changes.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# SemanticChunker groups sentences based on their meaning rather than
# fixed character count or paragraph boundaries.
#
# breakpoint_threshold_type="standard_deviation"
# ------------------------------------------------
# 1. Calculate semantic distance between consecutive sentences.
# 2. Compute the mean and standard deviation of those distances.
# 3. If a distance is unusually large (topic shift),
#    a chunk boundary is inserted.
#
# breakpoint_threshold_amount=1
# ------------------------------------------------
# A sentence pair is more likely to be separated if its semantic
# distance exceeds approximately 1 standard deviation from the average.
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

# Sample text containing three different topics:
# 1. Farming
# 2. IPL Cricket
# 3. Terrorism
#
# Since the topics are unrelated, SemanticChunker should ideally
# create around 3 chunks.
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for
the next season. The sun was bright, and the air smelled of earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world. People
all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates
fear in cities and villages. When such attacks happen, they leave behind pain and
sadness. To fight terrorism, we need strong laws, alert security forces, and support
from people who care about peace and safety.
"""

# Convert the text into LangChain Document objects.
# SemanticChunker will analyze sentence meanings and determine
# where topic changes occur.
docs = text_splitter.create_documents([sample])

# Print the number of generated chunks.
print(f"Number of chunks: {len(docs)}")

# Display each chunk separately.
for i, doc in enumerate(docs, start=1):
    print(f"\n{'=' * 50}")
    print(f"Chunk {i}")
    print(f"{'=' * 50}")
    print(doc.page_content)