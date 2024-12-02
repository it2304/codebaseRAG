import streamlit as st
from groq import Groq
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
import tempfile
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone

SUPPORTED_EXTENSIONS = [".py", ".js", ".tsx", ".ts", ".java", ".cpp"]
IGNORED_DIRS = ["node_modules", ".git", "dist", "__pycache__", ".next", ".vscode", "env", "venv"]

def test_input_link(repo_link):
    # Extract owner and repo name from the link
    try:
        # Example repo_link: https://github.com/owner/repo
        parts = repo_link.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
    except (IndexError, ValueError):
        return False

    # GitHub API URL for repository details
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Send a GET request to the GitHub API
    response = requests.get(api_url)

    # Return True if the status code indicates success (200), otherwise False
    return response.status_code == 200

def clone_repo(repo_url):
  repo_name = repo_url.rsplit("/", 1)[-1]
  repo_path = f"/{repo_name}"
  Repo.clone_from(repo_url, str(repo_name))
  return str(repo_path)

def get_file_content(file_path, repo_path):
  try:
    with open(file_path, "r", encoding="utf-8") as f:
      content = f.read()
      rel_path = os.path.relpath(file_path, repo_path)

      return {
          "name": rel_path,
          "content": content
      }
  except Exception as e:
    print(f"Error reading file {file_path}: e")
    return None

def get_main_files_content(repo_path: str):
    files_content = []

    try:
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
  model = SentenceTransformer(model_name)
  return model.encode(text)

def update_pinecone(file_content, repo_url):
    documents = []

    for file in file_content:
        doc = Document(
            page_content=f"{file['name']}\n{file['content']}",
            metadata={"source": file['name']}
        )

        documents.append(doc)

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(),
        index_name="codebase-rag",
        namespace=repo_url
    )

if __name__ == "__main__":
    st.title("Codebase RAG")

    # Set OpenAI API key from Streamlit secrets
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    # Set up pinecone client
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index("codebase-rag")

    repo_url = ""

    # Set a default model
    if "llama_model" not in st.session_state:
        st.session_state["llama_model"] = "llama-3.1-70b-versatile"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize selected links tracking
    if "selected_links" not in st.session_state:
        st.session_state.selected_links = set()

    col1, col2 = st.columns(2)

    # Set up a column to receive repo links
    with col1:
        st.header("Input Links")

        # Initialize link history if not already in session state
        if "link_history" not in st.session_state:
            st.session_state.link_history = []

        link_input = st.text_input("Enter a link:")

        # Add a button next to the input field using columns for better layout control
        col1_1, col1_2 = st.columns([3, 1])

        with col1_2:
            if st.button("Embed"):
                if test_input_link(link_input):
                    st.session_state.link_history.append(link_input)  # Add to history
                    path = clone_repo(link_input)
                    file_content = get_main_files_content(path)
                    embeddings = get_huggingface_embeddings(file_content)
                    update_pinecone(file_content, link_input)
                else:
                    st.error("Please enter a valid link before embedding.")


        if st.session_state.link_history:
            # Display links with checkboxes
            for idx, link in enumerate(st.session_state.link_history):
                col1_1, col1_2 = st.columns([1, 10])

                with col1_1:
                    checked = st.checkbox("", key=f"check_{idx}", value=(link in st.session_state.selected_links))
                    if checked:
                        st.session_state.selected_links.add(link)
                    else:
                        st.session_state.selected_links.discard(link)

                with col1_2:
                    st.markdown(f"[{link}]({link})")  # Render link as clickable markdown

            # Show selected links if any
            if st.session_state.selected_links:
                st.subheader("Selected Links")
                for selected_link in st.session_state.selected_links:
                    st.markdown(f"✅ [{selected_link}]({selected_link})")
                    repo_url = selected_link

    # Set up a column to interact with codebase
    with (col2):
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            query_embedding = get_huggingface_embeddings(prompt)
            top_matches = pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=3,
                include_metadata=True,
                namespace=repo_url
            )
            contexts = [item['metadata']['text'] for item in top_matches['matches']]
            augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[: 10]) \
                              + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + prompt

            with st.chat_message("system"):
                system_prompt = f"""You are a Senior Software Engineer, 
                with over 10 years of experience in TypeScript and Python.

                Answer any questions I have about the codebase, based on the code provided. 
                Always consider all of the context provided when forming a response.

                Let's think step by step. Verify step by step.
                """

                llm_response = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": augmented_query}
                    ]
                )

                response = st.write(llm_response.choices[0].message.content)
            st.session_state.messages.append({"role": "system", "content": response})