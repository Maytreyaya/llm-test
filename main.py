from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()


def create_db_from_pdf(pdf_file_path: str) -> FAISS:
    # Load PDF content
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    # Create database using embeddings
    db = FAISS.from_documents(docs, embeddings)

    return db


# Function to get response from a query
def get_response_from_query(db, query, k=4):
    # Search the database for similar documents
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize OpenAI language model
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions based on the PDF files.

        Answer the following question: {question}
        By searching the following document: {docs}

        Only use the factual information from the document to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """,
    )

    # Create LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate response
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


pdf_file_path = "C:/Users/maksy/projects/llm-test/CoffeeB_Manual Globe_EN_10.08.2022.pdf"

# Create database from PDF
db = create_db_from_pdf(pdf_file_path)

# Query for response
query = 'How can I resolve the issue of my coffee being trapped in the machine?'
response, related_documents = get_response_from_query(db, query)

# Print the response
print("Response:", response)
