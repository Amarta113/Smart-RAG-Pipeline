import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import qdrant_client
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from style import css, bot_template, user_template

load_dotenv()

def get_vector_store():
    web_loader = WebBaseLoader(["https://www.gaditek.com/","https://www.gaditek.com/our-impact/",
                                "https://www.gaditek.com/careers/","https://www.gaditek.com/benefits/",
                                "https://www.linkedin.com/company/gaditek/","https://www.linkedin.com/in/arqamgadit/",
                                "https://www.linkedin.com/in/umairgadit/?originalSubdomain=ae","https://www.linkedin.com/company/wearedisrupt/",
                                "https://disrupt.com/"
                                ]                
        )

    pages = web_loader.load_and_split()
    
    txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex=False
    )

    
    doc_list = []

    for page in pages:
        pag_split = txt_splitter.split_text(page.page_content)

        for pag_sub_split in pag_split:
            metadata = {"source": "Website", "page_no": page.metadata.get('page', 0) + 1}
            doc_string = Document(page_content= pag_sub_split, metadata=metadata)
            doc_list.append(doc_string)

    model_name = 'BAAI/bge-small-en'
    embed_model = HuggingFaceBgeEmbeddings(model_name=model_name)

    vector_store = QdrantVectorStore.from_documents(
                        doc_list,
                        embed_model,
                        url=os.getenv("qdrant_url"),
                        api_key=os.getenv("qdrant_key"),
                        collection_name='afiniti'

                        )

    return vector_store


def get_context_retriever_chain(vector_store, api_key):
    llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0, groq_api_key=api_key)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_converstional_rag_chain(retriever_chain, api_key):
    llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0, groq_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_docs_chain)

def get_response(user_query, vector_store, api_key):
    retriever_chain = get_context_retriever_chain(vector_store, api_key)
    conversation_rag_chain = get_converstional_rag_chain(retriever_chain, api_key)
    response = conversation_rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_query
           })
            
    return response['answer']

st.set_page_config(page_title="Gaditech", page_icon=":cyclone:")
st.write(css, unsafe_allow_html=True)

st.title("Gaditech AI Assistant")

with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Groq API key")

if api_key is None or api_key == "":
    st.info("Please enter your groq api key")


else:       
    # session state
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am Gaditech AI assistant. Ask me about Gaditech.")
            ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()
        
    vector_store = st.session_state.vector_store
        
    # User Input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
            response = get_response(user_query, vector_store, api_key)      
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:    
        if isinstance(message, AIMessage):
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

