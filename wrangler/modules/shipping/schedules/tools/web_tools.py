#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# import requests, os
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper

class searchWorkLoads():

    def __init__(self) -> None:
        return None

    def duckduckgo(self)-> Tool:

        # Initialize DuckDuckGo Search
        search = DuckDuckGoSearchAPIWrapper()

        return Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for searching the internet for recent information."
        )


#     @tool("Topic content read and write Tool")
#     def topics_search(query: str):
#         """Fetch topic specific content and process them."""
# #         API_KEY = os.getenv('NEWSAPI_KEY')  # Fetch API key from environment variable
# #         base_url = "https://newsapi.org/v2/everything"
        
#         params = {
#             'q': query,
# #             'sortBy': 'publishedAt',
# #             'apiKey': API_KEY,
#             'language': 'en',
#             'pageSize': 5,
#         }
        
#         response = requests.get(base_url, params=params)
#         if response.status_code != 200:
#             return "Failed to retrieve topic content."
        
#         articles = response.json().get('articles', [])
#         all_splits = []
#         for article in articles:
#             # Assuming WebBaseLoader can handle a list of URLs
#             loader = WebBaseLoader(article['url'])
#             docs = loader.load()

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             splits = text_splitter.split_documents(docs)
#             all_splits.extend(splits)  # Accumulate splits from all articles

#         # Index the accumulated content splits if there are any
#         if all_splits:
#             vectorstore = Chroma.from_documents(all_splits, 
#                                                 embedding=embedding_function, 
#                                                 persist_directory="./chroma_db")
#             retriever = vectorstore.similarity_search(query)
#             return retriever
#         else:
#             return "No content available for processing."