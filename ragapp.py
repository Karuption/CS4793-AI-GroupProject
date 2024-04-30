from llama_index.readers.web import NewsArticleReader
from llama_index.core.schema import Document

# Scrape the link and return a Document if found or a None if not found
# Using LLama-Index since it has a better fetching system
def get_article(link: str) -> Document|None:
    article_link = link
    article = NewsArticleReader(html_to_text=True).load_data([article_link])
    if len(article) > 0:
        return article[0]
    else:
        return None

from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

print("Setting up Embeddings")
# Make the Model used for Embeddings
# nomic-embed-text needs to be ran by Ollama locally
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    temperature=0.1,
    show_progress=True,
    num_thread=16,
)

print("Loading Vector Store")
# Get the vector store that was produced from the RAG Jupiter notebook
vectorstore = Chroma(
    persist_directory="./chroma", 
    embedding_function=embeddings
)

print("Getting Retriever")
# Take the vector store and make it into a retriever so that it can be queried
retriever = vectorstore.as_retriever(
    search_type="similarity",
)

from langchain_community.llms.ollama import Ollama

print("setting up the LLM connection")
# Set up the LLM from the Ollama instance running Mistral
llm = Ollama(
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.1,
    num_thread=16,
)

# Format the docs got from the retriever
def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Template to use for the LLM to generate on
# Context will be provided from the retriever and question is the rag prompt
template = """Use the following pieces of context to to generate a short sentence based on the article provided.
Keep the tweet as concise, sparse, and close to the context's provided language use as possible.

{context}

Article: {question}

Short Sentence:"""
custom_rag_prompt = PromptTemplate.from_template(template)

print("Making RAG chain")
# Set up the RAG to output text. Context is getting related docs and question is the rag prompt passthrough
rag_chain = (
    {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

from flask import Flask, Response, send_file, request
from flask_sse import sse

# Set up flask for an API
app = Flask("RAG")

# default route, serve the index page
@app.route('/', methods = ['GET'])
async def index():
    return send_file('index.html')    

# This is a form submission route. 
# articleTextbox is the name of the text field in question
@app.route('/ragRequest', methods = ['POST'])
async def tweet_generator():
    article_link = request.form.get('articleTextbox')

    # Return an HTMX powered response that will go to the correct route and stream the result
    return f"""
        <!-- This is htmx and sse powered. This will automatically try to connect to the sse-connect url -->
        <!-- This div is listening to the "completed" event. That event will swap out the connection to prevent from reconnecting-->
        <div hx-ext="sse" sse-connect="/ragResponse/{article_link}" sse-swap="completed" hx-swap="outerHTML">
            <!-- This div inherits the connection from its outer div. It will swap itself out with any message.-->
            <!-- The inner span gives a spinner that is powered by bootstrap -->
            <div sse-swap="message" hx-swap="outerHTML">
                <span class="spinner-border spinner-border-sm"></span>
            </div>
            </div>
        </div>"""

import time
# This is what scrapes and streams the result to the page
@app.route("/ragResponse/<path:url>", methods=['GET'])
def publish_hello(url=None):
    def stream(url):
        # Set up the main prefix. The "data :" is what tells HTMX it's an SSE
        prefix = f"data: <p sse-swap=\"message\">"
        msg = ""
        # \n\n is required by the SSE spec to be registered as an SSE
        end = "</p>\n\n"

        article = None
        if url != None:
            # Scrape the article
            article = get_article(url)

        # If there was no url or article, return a 'styled' error
        # Send as a event "completed" so that the client doesn't reconnect 
        if article == None:
            yield "event: completed\ndata: <p style=\"color: red;\">Bad URL" + msg + end

        # Iterate the response stream and pass in the article's summary
        # Summaries as input were observed to lead to give more "The Telegram" like tweets 
        for response in rag_chain.stream(article.metadata['summary']):
            msg += response
            yield prefix + msg + end
            # This slow down helps the api look more like ChatGPT style rather than a burst load
            time.sleep(.15)
        # Once the stream stops, send it again as a completed event so that the client doesn't reconnect
        yield "event: completed\ndata: <p>" + msg + end

    # open a connection for sse
    return Response(stream(url), mimetype="text/event-stream")
    
app.run(debug=True)