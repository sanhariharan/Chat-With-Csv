import streamlit as st
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

st.title("Chat with CSV using Llama2 🦙🦜")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    # Clear previous session state and vectorstore if any
    st.session_state['history'] = []
    st.session_state['generated'] = []
    st.session_state['past'] = []

    # Use tempfile because CSVLoader only accepts a file_path
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        data = None

    if data:
        try:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                               model_kwargs={'device': 'cpu'})

            db = FAISS.from_documents(data, embeddings)

            # Ensure vectorstore directory exists
            os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
            db.save_local(DB_FAISS_PATH)

            llm = load_llm()
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
        except Exception as e:
            st.error(f"Error creating embeddings or vectorstore: {e}")
            chain = None

        if chain:
            # Initialize session state keys before defining conversational_chat
            if 'history' not in st.session_state:
                st.session_state['history'] = []
            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]
            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey! 👋"]

            def conversational_chat(query):
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
                result = chain({"question": query, "chat_history": st.session_state['history']})
                st.session_state['history'].append((query, result["answer"]))
                return result["answer"]

            # Container for the chat history
            response_container = st.container()
            # Container for the user's text input
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    output = conversational_chat(user_input)

                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        st.markdown(f"**User:** {st.session_state['past'][i]}")
                        st.markdown(f"**Bot:** {st.session_state['generated'][i]}")
