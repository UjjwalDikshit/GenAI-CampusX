# Don't do data validation with help of TypeDict, it may malfunction, It is better for representation purpose
# for data validatioin use Pydantic -> data validation and data parsing library in python.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional # optional tell it is not required to give by llm
from typing import Literal # help to decide result from within your options by llm pos,neg (removed neutral)

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# create schema for output like what i looks

# -------------------------------------------------------
# simple 

# class Review(TypedDict):
#     summary:str
#     sentiment:str

# -------------------------------------------------------
# may be sometime llm doesn't understand with keppoint that what i have to genrate, so we'll give some description for each keyword. For this Use Annotated
# Literal gives option to llm, and answer have to be from them
# Optional tell whether llm can return or not that attribute.

class Review(TypedDict):
    key_themes:Annotated[list[str],"Write all the key themes discussed in the review"]
    summary:Annotated[str,"A brief summary of the review"]
    # sentiment:Annotated[str,"Return setiment of the reiew either negative, positive, neutral"]
    sentiment:Annotated[Literal["pos","neg"],"Return setiment of the reiew either negative, positive, neutral"] # choose from pos, neg only
    pros:Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons:Annotated[Optional[list[str]],"Write down all the cons inside a list"]

# -------------------------------------------------------


# pass| tell this structure to model for ouput 

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3
processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x
actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard
pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
""")

# print(result)
# print(result['summary'])
print(result['sentiment'])

# -------------------------------------------------------
#pydantic version

# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )

# class Review(BaseModel):
#     summary: str
#     sentiment: str

# structured_model = model.with_structured_output(Review)

# result = structured_model.invoke(
#     """The hardware is great, but the software feels bloated.
#     There are too many pre-installed apps that I can't remove.
#     Also, the UI looks outdated compared to other brands.
#     Hoping for a software update to fix this."""
# )

# print(result)
# print(result.summary)
# print(result.sentiment)