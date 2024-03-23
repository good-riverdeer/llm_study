from dotenv import load_dotenv
import streamlit as st

from backend.vector_store import load_repository, store_repository
from backend.agents import response_from_retrieval_chain


load_dotenv()

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "docs" not in st.session_state:
    st.session_state["docs"] = None

st.title("Ollama Inference")

st.info(
    """Chat with sLLM like **codellama:7b** with contexts from your own repository!

* Generate RAG, following instructions below.
* **Github repo name** - Type public github repo URL to generate context.
  * format: `user_name/repo_name`
  * example: `huggingface/transformers`
* **Branch name** - Type a branch name to generate context from. 
* **File extension** - Select file extension to generate context for.
  * If you don't know, try `.py` or `.md`.
* **File filter** - """
)

with st.sidebar:
    model_name = st.selectbox(
        "Model_name",
        [
            "codellama:7b",
        ],
    )
    repo_name = st.text_input("Github Repo Name", "langchain-ai/langchain")
    branch = st.text_input("Branch name", "master")
    file_extention = st.selectbox("File extension", [".md", ".py"])
    file_filter = st.text_input("File filter", "libs/")

    if st.session_state.docs:
        st.write("Here are files that I know!")
        with st.status("Wait for storing vectors"):
            for doc in st.session_state.docs:
                st.write(f"* {doc.metadata['path']}")

    if st.button("Submit"):
        with st.status("Wait for generating context"):
            st.session_state.docs = None
            docs = []
            docs_stream = load_repository(
                repo_name,
                branch=branch,
                file_extention=file_extention,
                file_filter=lambda x: file_filter in x,
            )
            for doc in docs_stream:
                docs.append(doc)
                st.write(f"* {doc.metadata['path']}")
            db_store = repo_name.replace("/", "_")
            db = store_repository(
                docs=docs, model_name=model_name, collection_name=f"{db_store}_store"
            )
            retriever = db.as_retriever()
            st.session_state.retriever = retriever
        st.write("Done!")
        st.write(f"You can now use the model {model_name} to generate text!")
        st.session_state.docs = docs

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("New Session"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.retriever:
        with st.chat_message("assistant"):
            st.error(
                "First, you need to store the documents in the repository!"
                "please chek **Sidebar** on your left side."
            )
    else:
        with st.chat_message("assistant"):
            stream = response_from_retrieval_chain(
                model_name,
                prompt,
                st.session_state.retriever,
                st.session_state.messages,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
