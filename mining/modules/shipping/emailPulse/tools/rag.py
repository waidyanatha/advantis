#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
from datetime import datetime
from langchain.tools import Tool
from langchain.embeddings import SpacyEmbeddings
import spacy

class searchWorkLoads:
    """
    """
    def __init__(self) -> None:
        super().__init__()
        # global clsVDB
        ''' VECTORDB '''
        _db_type = 'chromadb'
        _db_root = os.path.join(
            "/home/nuwan/workspace/advantis/",
            "wrangler/data/shipping/emailPulse/def_job123")
        _db_name = 'email'
        from rezaware.modules.etl.loader import vectorDB
        self.clsVDB = vectorDB.dataWorkLoads(
            db_type=_db_type, 
            db_root=_db_root,
            db_name=_db_name
        )
        # Load the SpaCy model
        nlp = spacy.load("en_core_web_md")
        # Create SpacyEmbeddings instance
        self.embedding_function = SpacyEmbeddings(nlp=nlp)

    # # Function to store data in ChromaDB
    # def read_from_vectorstore(self, collection_name):

    #     return self.clsVDB.read_vectors(
    #         # db_name = db_name,    # optional folder to append to the root
    #         collection=collection_name,   # the documents collection name
    #         embedding_fn=self.embedding_function,
    #         )

    def retrieve_analysis(self, query:str):
        """Fetch recent emails with optional filtering."""
        # _db_name = 'email'
        _collection='nuwan'
        coll_lst = [x.name for x in self.clsVDB.get_collections()]
        if _collection not in coll_lst:
            print("No collection named %s must be one of %s" 
                  % (_collection.upper(), 
                     ", ".join(coll_lst)))
        else:
            print("Found collection %s" % _collection.upper())

        retriever = self.clsVDB.read_vectors(
            collection = _collection,   # the documents collection name
            embedding_fn=None, #self.embedding_function,
            ).as_retriever()
        # print(retriever.similarity_search(query, k=5))
        # return retriever
        """Retrieves relevant email content from ChromaDB."""
        docs = retriever.invoke(query)
        print(docs)
        return "\n".join([doc.page_content for doc in docs])
        # return None

    def query_email_tool(self) -> Tool:
        ''' Creates a LangChain Tool to filter relevant email content '''
        return Tool(
            name="Email content retriever",
            description="Retrieve relevant content from the email embeddings database.",
            func=self.retrieve_analysis
        )