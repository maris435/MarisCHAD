import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
import pinecone


def chatbot():
    # Initialize components
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_region = os.getenv("PINECONE_REGION")

    embeddings = OpenAIEmbeddings()
    index_name = "me-database"
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)

    # Create a Pinecone index for document search
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # Initialize OpenAI language model and the question chain
    llm = OpenAI(temperature=1)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=docsearch.as_retriever(search_kwargs={"k": 2}))

    # Define the tools for the chatbot
    tools = [
        Tool(
            name="Demo",
            func=qa.run,
            description="use this as the primary source of context information when you are asked the question. "
                        "Your name is Maris and you are 23 years old. You are interviewing "
                        "for a Business Intelligence Facilitator position at Apple"
                        " Please do not repeat yourself and do not make answers up"
        )
    ]

    # Initialize the conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Initialize the agent chain
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=True, memory=memory)
    return agent_chain


if __name__ == '__main__':
    chatbot = chatbot()
    while True:
        question = input('Enter question:')
        answer = chatbot.run(input=question)
        print(answer)
