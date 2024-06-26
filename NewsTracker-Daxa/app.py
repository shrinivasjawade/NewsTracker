# Import necessary libraries and modules
from flask import Flask, request, render_template, jsonify                      # Flask web framework components
import requests                                                                 # For making HTTP requests
from bs4 import BeautifulSoup                                                   # For web scraping
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline      # For BERT model and pipeline
from langchain_community.vectorstores import Chroma                             # For vector database
from langchain_community.embeddings import OllamaEmbeddings                     # For embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter             # For text splitting
from langchain.prompts import ChatPromptTemplate, PromptTemplate                # For creating prompts
from langchain_core.output_parsers import StrOutputParser                       # For output parsing
from langchain_community.chat_models import ChatOllama                          # For chat models
from langchain_core.runnables import RunnablePassthrough                        # For running tasks
from langchain.retrievers.multi_query import MultiQueryRetriever                # For multi-query retrieval
import nltk                                                                     # Natural Language Toolkit
from nltk.corpus import stopwords                                               # For stopwords
from nltk.tokenize import word_tokenize                                         # For tokenizing words

#################################################################################################################################

# Download stopwords and punkt tokenizer models
nltk.download('stopwords')
nltk.download('punkt')

#################################################################################################################################

# Initialize the Flask application
app = Flask(__name__)

#################################################################################################################################

# Function to remove stopwords from text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    return filtered_text

# Function to scrape and clean text from a webpage
def scrape_search_results(url):
    response = requests.get(url)                                              # Fetch the webpage content
    soup = BeautifulSoup(response.text, 'html.parser')                        # Parse the HTML content
    text = ' '.join([p.get_text() for p in soup.find_all('p')])               # Extract text from paragraph tags
    text = remove_stopwords(text)                                             # Remove stopwords from the text
    return text

# Define a class to handle documents
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

#################################################################################################################################

# Load BERT QA model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Initialize the QA pipeline
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Initialize the text splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
vector_db = None                                                            # Placeholder for the vector database

# Initialize the local LLM (Large Language Model) from Ollama
local_model = "llama3"
llm = ChatOllama(model=local_model)

#################################################################################################################################

# Define a prompt template for generating alternative questions
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = None                                                        # Placeholder for the retriever
chain = None                                                            # Placeholder for the QA chain

#################################################################################################################################

@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form['url']                                           # Get the URL from the form
    scraped_text = scrape_search_results(url)                           # Scrape and clean the article text
    document = Document(scraped_text)                                   # Create a Document object

    global vector_db
    chunks = text_splitter.split_documents([document])                  # Split the document into chunks

    # Create a vector database from the document chunks
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )

    global retriever, chain
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # Define a prompt template for the QA system
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Define the QA chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Define a prompt template for summarizing the text
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize the following text in a concise manner:\n{text}"""
    )

    # Define the summary chain
    summary_chain = (
        {"text": RunnablePassthrough()}
        | summary_prompt
        | llm
        | StrOutputParser()
    )

    # Generate the summary
    summary = summary_chain.invoke(input=document.page_content)
    return jsonify(summary=summary)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']                                 # Get the question from the form
    if question.lower() in ["exit", "quit"]:
        return jsonify(answer="Chatbot session ended.")
    answer = chain.invoke(question)                                     # Get the answer using the QA chain
    return jsonify(answer=answer)

#################################################################################################################################

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)