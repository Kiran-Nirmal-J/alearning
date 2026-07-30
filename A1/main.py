from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Step 1  Simple prompt implementation
messages = [
    SystemMessage(content="You are a helpful assistant that answers in one line."),
    HumanMessage(content="Who are you?"),
]


response = llm.invoke(messages)
print(response.content)

# invoke call implementation
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {domain} expert. Be consise and answer in one line."),
        ("human", "Explain {concept} in simple terms."),
    ]
)

chain = prompt_template | llm | StrOutputParser()
result = chain.invoke({"domain": "finance", "concept": "inflation"})
print(result)

# batch call implementation
batch_inputs = [
    {"domain": "finance", "concept": "inflation"},
    {"domain": "mathematics", "concept": "linear equations"},
    {"domain": "physics", "concept": "Newton's laws of motion"},
]
batch_results = chain.batch(batch_inputs)
for i, result in zip(batch_inputs, batch_results):
    print(f"\n {i['domain']} - {i['concept']}: {result}")


# pydantic implementation
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: str = Field(description="Rating from 1 to 10")


structured_llm = llm.with_structured_output(MovieReview)
review_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic. Provide a review in JSON format."),
        ("human", "Write a review for the movie '{title}'"),
    ]
)
review_chain = review_prompt | structured_llm
review_result: MovieReview = review_chain.invoke({"title": "The Odessey"})
print(f"\nMovie Title: {review_result.title}")
print(f"Rating: {review_result.rating}")
