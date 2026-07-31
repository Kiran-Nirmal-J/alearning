import sqlite3

from dotenv import load_dotenv

load_dotenv()

from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
LINE = "^" * 50


# Task 1 : Build graph and node
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    # print("State messages:", state["messages"])
    answer = llm.invoke(state["messages"])
    return {"messages": [answer]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

conn = sqlite3.connect("assignment_state.db", check_same_thread=False)
_db = SqliteSaver(conn)
graph = builder.compile(checkpointer=_db)

print(LINE)
print("TASK 1 output")
print(graph.get_graph().draw_ascii())

# Task 2 : Memory check
conf1 = {"configurable": {"thread_id": "1"}}
print(LINE)
print("TASK 2 output")
print(LINE)
r = graph.invoke({"messages": [HumanMessage(content="Hi, I'm Kiran.")]}, config=conf1)
print("Turn 1:", r["messages"][-1].content)

r = graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=conf1)
print("Turn 2 :(memory check)", r["messages"][-1].content)
print(LINE)
### printing the output
# Turn 1: Hi, Kiran! How can I help you today?
# Turn 2 :(memory check) Your name is Kiran.

### TASK - 3 With new thread_id
conf2 = {"configurable": {"thread_id": "2"}}
r2 = graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]}, config=conf2
)
print(LINE)
print("TASK 3 output")
print(LINE)
print("Thread 2 answer:", r2["messages"][-1].content)
print(LINE)

### printing the output
# Thread 2 answer: I'm sorry, but I don't have access to personal information about users unless it has been shared with me in the course of our conversation. If you'd like to tell me your name, feel free!

print(LINE)
print("""
SECURITY REFLECTION - Why isolation is a security concern, not just a feature.
Isolation is a security concern because it ensures that different threads or sessions do not share sensitive information inadvertently
Unlike a stateless api where the worst outcome is a wrong answer, in a stateful api, the worst outcome is a data leak.
""")
print(LINE)

print(LINE)
print("TASK 4 Inspect and mutate thread 1 state")
print(LINE)
snapshot = graph.get_state(conf1)
print("\nStored messages for thread 1 ({len(snapshot['messages'])} total):")
for m in snapshot.values["messages"]:
    print(f"type={m.type:<12} content={m.content}")

# mutate
graph.update_state(conf1, {"messages": [HumanMessage(content="I like cricket.")]})
snapshot_after = graph.get_state(conf1)
print("\nStored messages for thread 1 ({len(snapshot_after['messages'])} total):")
for m in snapshot_after.values["messages"]:
    print(f"type={m.type:<12} content={m.content}")


print(LINE)
print("TASK 5 Trimming the messages")
print(LINE)

raw_history = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi, I'm Kiran."),
    AIMessage(content="Hi, Kiran! How can I help you today?"),
    HumanMessage(content="I am learning coding"),
    AIMessage(content="That's great! What programming languages are you learning?"),
    HumanMessage(content="I am learning Python and JavaScript."),
    AIMessage(
        content="Python and JavaScript are both excellent choices! Do you have any specific projects in mind that you'd like to work on?"
    ),
    HumanMessage(content="I want to build a web application."),
    AIMessage(
        content="Building a web application is a great way to apply your coding skills. Do you have any particular ideas for the application?"
    ),
    HumanMessage(content="I want to build a task management app."),
    AIMessage(
        content="A task management app is a useful project. You can start by defining the features you want, such as creating tasks, setting deadlines, and tracking progress. Do you need help with the design or the coding part?"
    ),
    HumanMessage(content="I need help with the coding part."),
    AIMessage(
        content="Sure! For a task management app, you can use a web framework like Flask or Django for Python, or Express.js for JavaScript. You'll need to set up routes, create a database to store tasks, and implement the front-end interface. Do you want to start with the back-end or the front-end?"
    ),
]


trimmer = trim_messages(
    max_tokens=100,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

surviving = trimmer.invoke(raw_history)
print(f"Length of Original messages: {len(raw_history)}")
print(f"Length of Surviving messages after trimming: {len(surviving)}")
for m in surviving:
    label = f"[{m.type}]"
    print(f"{label:<12} content={m.content}")


print(LINE)
print("Reflection answers")

print("""
What exactly does the add_messages reducer do, and what would break without it?
Answer : The add_messages reducer is responsible for adding new messages to the existing list of messages in the state. Without it, new messages would not be properly incorporated into the conversation history.

2. What does attaching a checkpointer change about how invoke behaves?
Answer : Attaching a checkpointer allows the state of the conversation to be saved and retrieved across different invocations. This means that the conversation can maintain context and continuity, even if the application is restarted or if different threads are used.

3. Give one reason you'd trim history beyond just fitting the context window.
Answer : Trimming history can help improve performance by reducing the amount of data that needs to be processed, which can lead to faster response times and lower resource usage. It can also help maintain relevance by focusing on the most recent and pertinent parts of the conversation.
""")
print(LINE)
