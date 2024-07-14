import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from openai import OpenAI

# Function to extract text from PDF
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Create vector store from chunks
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Main Streamlit app function
def main():
    load_dotenv()
    st.set_page_config(page_title="AI Job Application Assistant", page_icon=":briefcase:", layout='wide')
    st.title("AI Job Application Assistant")
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Sidebar for PDF upload
    with st.sidebar:
        docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if docs and st.button("Process Documents"):
            raw_text = get_pdf_text(docs)
            text_chunks = get_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state['conversation_chain'] = ConversationalRetrievalChain(
                llm=client,
                retriever=vectorstore.as_retriever(),
                memory=ConversationBufferMemory(memory_key='chat_history')
            )

    # Handle chat stream
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.messages) >= 20:  # Example limit for messages
        st.info("Maximum message limit reached.")
    else:
        query = st.chat_input("What is up?")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            try:
                # Sending the query to the model
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True
                )
                # Process the stream
                response = st.write_stream(stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(e)}
                )

if __name__ == '__main__':
    main()
