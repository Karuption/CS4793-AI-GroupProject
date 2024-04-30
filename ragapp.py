import nest_asyncio

# Needed for jupyter notebooks to work with async tasks
nest_asyncio.apply()

# Llama-index has a really easy to use news article parser. It takes a list of urls as strings
from llama_index.readers.web import NewsArticleReader
from llama_index.core.schema import Document

def get_article(link: str) -> Document:
    article_link = link
    article = NewsArticleReader(html_to_text=True).load_data([article_link])
    return article[0]

from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    temperature=0.1,
    show_progress=True,
    num_thread=16,
)

vectorstore = Chroma(
    persist_directory="./chroma", 
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
)

from langchain_community.llms.ollama import Ollama

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

rag_chain = (
    {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

from flask import Flask, send_file, request

app = Flask("RAG")

@app.route('/', methods = ['GET'])
async def index():
    return send_file('index.html')    


import time
@app.route('/RAG', methods = ['POST'])
async def tweet_generator():
    time.sleep(1)
    return """
    <div class="form-group">
        <textarea class="form-control input-bg" id="generatedText" rows="5" tabindex="-1"
            readonly>{auhetasuhetansh}</textarea>
    </div>"""
    article_link = request.form.get('articleTextbox')
    print("scraping article at " + article_link)
    article = get_article(article_link)
    print("Sending sending to rag chain")
    return await rag_chain.ainvoke(article.metadata['summary'])
    
app.run(debug=True)