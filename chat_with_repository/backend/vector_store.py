import os
import requests, base64
from typing import List, Dict, Iterator

from langchain_core.documents import Document
from langchain_community.document_loaders.github import GithubFileLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma


class CustomGithubFileLoader(GithubFileLoader):
    def get_file_paths(self) -> List[Dict]:
        base_url = (
            f"{self.github_api_url}/repos/{self.repo}/git/trees/"
            f"{self.branch}?recursive=1",
        )
        response = requests.get(*base_url, headers=self.headers)
        response.raise_for_status()
        all_files = response.json()["tree"]
        return [
            f
            for f in all_files
            if not (self.file_filter and not self.file_filter(f["path"]))
        ]

    def get_file_content_by_path(self, path: str) -> str:
        base_url = f"{self.github_api_url}/repos/{self.repo}/contents/{path}"
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()

        if isinstance(response.json(), dict):
            content_encoded = response.json()["content"]
            return base64.b64decode(content_encoded).decode("utf-8")

        return ""

    def load(self) -> List[Document]:
        documents = []
        files = self.get_file_paths()
        files = list(filter(lambda x: x["path"].endswith(self.file_extension), files))
        for file in files:
            content = self.get_file_content_by_path(file["path"])
            if content == "":
                continue

            metadata = {
                "path": file["path"],
                "sha": file["sha"],
                "source": f"{self.github_api_url}/repos/{self.repo}/{file['type']}/"
                f"{self.branch}/{file['path']}",
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def lazy_load(self) -> Iterator[Document]:
        files = self.get_file_paths()
        files = list(filter(lambda x: x["path"].endswith(self.file_extension), files))
        for file in files:
            content = self.get_file_content_by_path(file["path"])
            if content == "":
                continue

            metadata = {
                "path": file["path"],
                "sha": file["sha"],
                "source": f"{self.github_api_url}/repos/{self.repo}/{file['type']}/"
                f"{self.branch}/{file['path']}",
            }
            yield Document(page_content=content, metadata=metadata)


def load_repository(repo: str, **gl_kwargs):
    loader = CustomGithubFileLoader(
        access_token=os.environ.get("GITHUB_API_TOKEN"), repo=repo, **gl_kwargs
    )
    return loader.lazy_load()


def store_repository(docs: List[Document], model_name: str, collection_name: str):
    embeddings = OllamaEmbeddings(
        base_url=os.environ["OLLAMA_API_URL"],
        model=model_name,
    )
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return db
