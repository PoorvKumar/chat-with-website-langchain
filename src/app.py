import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstore_from_url(url,embeddings_model):
    loader=WebBaseLoader(url)
    document=loader.load()

    # split document into chunks
    text_splitter=RecursiveCharacterTextSplitter()
    document_chunks=text_splitter.split_documents(document)

    # create vector store from chunks
    vector_store=Chroma.from_documents(document_chunks,embeddings_model)
    return vector_store

def get_context_retriever_chain(vector_store,chat_model):
    # llm=ChatOpenAI()
    llm=chat_model()

    retriever=vector_store.as_retriever()

    prompt=ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain=create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain,chat_model):
    # llm=ChatOpenAI()
    llm=chat_model()

    prompt=ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}")
    ])

    stuff_documents_chain=create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuff_documents_chain)


def get_response(user_query,chat_model):
    # create conversation chain
    retriever_chain=get_context_retriever_chain(st.session_state.vector_store,st.session_state.chat_model)
    conversation_rag_chain=get_conversational_rag_chain(retriever_chain,st.session_state.chat_model)

    response=conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response["answer"]

#  app config
st.set_page_config(page_title="Chat with Websites",page_icon="🤖")
st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")
    model_selection = st.selectbox("Select Model", ["OpenAI", "GoogleGenerativeAI"])

if website_url is None or website_url=="":
    st.info("Please enter a website URl")
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[
            AIMessage(content="Hello, I am bot, How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        if model_selection == "OpenAI":
            st.session_state.embeddings_model = OpenAIEmbeddings()
            st.session_state.chat_model = ChatOpenAI
        elif model_selection == "GoogleGenerativeAI":
            st.session_state.embeddings_model = GoogleGenerativeAIEmbeddings()
            st.session_state.chat_model = ChatGoogleGenerativeAI
        st.session_state.vector_store=get_vectorstore_from_url(website_url,st.session_state.embeddings_model)

    # user input
    user_query=st.chat_input("Type your message here...")
    if user_query is not None and user_query!="":

        response=get_response(user_query,st.session_state.chat_model)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    #  conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
