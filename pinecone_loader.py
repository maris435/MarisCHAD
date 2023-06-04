import config
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pinecone

# Load the resume document using PyPDFLoader
loader = PyPDFLoader('reviewed_resume.pdf')

pages = loader.load()

# Initialize text splitter for splitting document into chunks
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)


# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

# Define an index name and initialize Pinecone with API key and environment

index_name = "me-database"
pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_REGION
)

# Split the document into individual pages or chunks
docs = text_splitter.split_documents(pages)

# Create a Pinecone index with the document chunks and embeddings
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# OR

# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)





