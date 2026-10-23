from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableSequence
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

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def main():
    
    video_id = "etcYBI7L1D8"

    transcript = find_transcript(video_id) 
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 100)

    chunks = splitter.create_documents([transcript]) 

    db = FAISS.from_documents(chunks,embeddings)

    retriever = db.as_retriever(
            search_type="mmr",                    
            search_kwargs={
                "k": 5,          # Number of final diverse documents you want to return
                "fetch_k": 20,     # Pool size to fetch initially for MMR re-ranking
                "lambda_mult": 0.5 # Balance between relevance (1) and diversity (0)
            } 
    )

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
    
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': RunnableSequence(retriever,RunnableLambda(format_docs)),
        'question':RunnablePassthrough()
    })

    sequential_chain = prompt | model | parser

    final_chain = parallel_chain | sequential_chain

    query  = "give me name of speaker and channel name, or any other information from which i can tract this video on youtbe and found it"
    query = "who is anna and jake"
    
    response = final_chain.invoke(query)

    print("Response: \n")

    print(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())