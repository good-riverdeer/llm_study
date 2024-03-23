from dotenv import load_dotenv
import streamlit as st

from backend.vector_store import load_arxiv_doc, store_docs
from backend.agents import response_from_retrieval_chain, summarizer


load_dotenv()

summary_template = """Hi! Finished my analysis. I'm ready to chat!
Here are some summarizations about the paper you gave me.
"""

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "docs" not in st.session_state:
    st.session_state["docs"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = None

st.title("Chat with Arxiv")
st.info("""Chat with LLM like **llama2:7b**""")

with st.sidebar:
    model_name = st.selectbox("Model", ["llama2:7b"])
    arxiv_query = st.text_input("Query", "2401.01055")

    if st.session_state.docs:
        st.write()
        metadata = st.session_state.docs[0].metadata
        st.info(f"Papers found!\n" f"* [{metadata['Title']}]({metadata['entry_id']})")

    if st.button("Search"):
        with st.spinner("Wait for generating context..."):
            st.session_state.docs = None
            docs, metadata = load_arxiv_doc(arxiv_query)

            db_store = metadata["entry_id"].split("/")[-1].replace(".", "_")
            db = store_docs(docs, model_name, db_store)
            retriever = db.as_retriever()

            st.session_state.retriever = retriever
            st.session_state.docs = docs

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.button("New Session"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.retriever is not None and st.session_state.summary is None:
    summary_stream = summarizer(model_name, metadata)
    with st.chat_message("assistant"):
        st.write(summary_template)
        summary = st.write_stream(summary_stream)
    st.session_state.summary = summary_template + summary
    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state.summary}
    )

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.retriever:
        with st.chat_message("assistant"):
            st.error(
                "First, you need to store the documents in Arxiv!"
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
