from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import sys

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

def find_transcript(video_id):
    try:
        # Create YouTube Transcript API instance
        api = YouTubeTranscriptApi()

        # Fetch English transcript
        transcript_list = api.fetch(
            video_id,
            languages=["en"]
        )

        # Extract transcript text
        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

        print("Transcript fetched successfully!\n")
        # print(transcript)
        print("\n")

        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video.")
        return None

    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def main():
    video_id = "P2GOkrU1vnQ"
    video_id = "J5_-l7WIO_w"
    video_id = "etcYBI7L1D8"

    transcript = find_transcript(video_id) # we can also have done this using document loader but that doen't work perfectly so using this.
    if transcript:
            print("\nTranscript length:\n", len(transcript))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 100)

    chunks = splitter.create_documents([transcript]) # break text in doc of chunks

    # what is role of id for each chunks, with each chunks you canseee embdedding and text

    db = FAISS.from_documents(chunks,embeddings)

    query  = "give me name of speaker and channel name, or any other information from which i can tract this video on youtbe and found it"

    # result = db.similarity_search(query, k = 1)  # don't do this instead use retrievers

    # print(db.get_by_ids(['52e61373-ace2-4f1a-a3b8-9fe8625eb6fc'])) # this doen't work, as everything we run the program, embedding store at differnt id's, so can't access it using
    # previous id, if you want to fetch usign previous ids then take help of google collab
    
    retriever = db.as_retriever(
            search_type="mmr",                    
            search_kwargs={
                "k": 5,          # Number of final diverse documents you want to return
                "fetch_k": 20,     # Pool size to fetch initially for MMR re-ranking
                "lambda_mult": 0.5 # Balance between relevance (1) and diversity (0)
            } 
    )

    # print(retriever.invoke(query))
    retrived_content = retriever.invoke(query)

    # moving to autmentation part

    prompt = PromptTemplate(
        template="""
            You are a helpful assitant.
            Answer only from provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            question: {question}
        """
        ,
        input_variables=['context','question']
    )

    context_text = "\n\n".join(doc.page_content for doc in retrived_content)

    # final_prompt = prompt.invoke({'context':context_text,'question':query})

    parser = StrOutputParser()

    chain  = prompt | model | parser
    response = chain.invoke({'context':context_text,'question':query})
    print("reponse: \n")
    print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())