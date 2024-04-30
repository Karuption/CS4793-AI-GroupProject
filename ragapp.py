from llama_index.readers.web import NewsArticleReader
from llama_index.core.schema import Document

def get_article(link: str) -> Document:
    article_link = link
    article = NewsArticleReader(html_to_text=True).load_data([article_link])
    if len(article) > 0:
        return article[0]
    else:
        return None

from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

print("Setting up Embeddings")
embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    temperature=0.1,
    show_progress=True,
    num_thread=16,
)

print("Loading Vector Store")
vectorstore = Chroma(
    persist_directory="./chroma", 
    embedding_function=embeddings
)

print("Getting Retriever")
retriever = vectorstore.as_retriever(
    search_type="similarity",
)

from langchain_community.llms.ollama import Ollama

print("setting up the LLM connection")
llm = Ollama(
    base_url="http://localhost:11434",
    model="mistral",
    temperature=0.1,
    num_thread=16,
)

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

template = """Use the following pieces of context to to generate a short sentence based on the article provided.
Keep the tweet as concise, sparse, and close to the context's provided language use as possible.

{context}

Article: {question}

Short Sentence:"""
custom_rag_prompt = PromptTemplate.from_template(template)

print("Making RAG chain")
rag_chain = (
    {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

from flask import Flask, Response, send_file, request
from flask_sse import sse

app = Flask("RAG")

@app.route('/', methods = ['GET'])
async def index():
    return send_file('index.html')    

@app.route('/ragRequest', methods = ['POST'])
async def tweet_generator():
    article_link = request.form.get('articleTextbox')
    return f"""
    <div id="generatedText" class="px-1 pt-2 pb-0">
        <div hx-ext="sse" sse-connect="/ragResponse/{article_link}" sse-swap="completed" hx-swap="outerHTML">
            <div sse-swap="message" hx-swap="outerHTML">
                <span class="spinner-border spinner-border-sm"></span>
            </div>
            </div>
        </div>
    </div>"""

import time
@app.route("/ragResponse/<path:url>", methods=['GET'])
def publish_hello(url=None):
    def stream(url):
        prefix = f"data: <p sse-swap=\"message\">"
        msg = ""
        end = "</p>\n\n"

        if url == None:
            yield "event: completed\ndata: <p style=\"color: red;\">Bad URL" + msg + end
        
        article = get_article(url)
        if article == None:
            yield "event: completed\ndata: <p style=\"color: red;\">Bad URL" + msg + end

        for response in rag_chain.stream(article.metadata['summary']):
            msg += response
            yield prefix + msg + end
            time.sleep(.15)
        yield "event: completed\ndata: <p>" + msg + end

    return Response(stream(url), mimetype="text/event-stream")
    
app.run(debug=True)