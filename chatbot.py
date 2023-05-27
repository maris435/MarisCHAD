# import config
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from boto.s3.connection import S3Connection
import pinecone
import os


def chatbot():
    # Initialize components
    os.environ["OPENAI_API_KEY"] = S3Connection(os.environ['OPENAI_API_KEY'], os.environ['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'])
    embeddings = OpenAIEmbeddings()
    index_name = "me-database"
    pinecone.init(api_key= S3Connection(os.environ['PINECONE_API_KEY'], os.environ['sk-5c877f11-4d28-4075-8938-d6e11d0da90d']),
                  environment=S3Connection(os.environ['PINECONE_REGION'], os.environ['us-east-1-aws']))

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