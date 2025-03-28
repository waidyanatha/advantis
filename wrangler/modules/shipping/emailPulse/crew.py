#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "crew"
__package__ = "emailPulse"
__module__ = "shipping"
__app__ = "wrangler"
__ini_fname__ = "app.ini"
__conf_fname__ = "app.cfg"

try:
    import os
    import sys
    import configparser    
    import logging
    import functools
    import traceback
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    from typing import List, Iterable, Dict, Tuple
    import litellm
    ''' CREWAI '''
    from crewai import Crew, Process #BaseAgent
    # from crewai import Crew, Agent, Task, Process, BaseAgent
    from crewai.project import CrewBase, crew
    # ''' caching '''
    # from langchain_core.globals import set_llm_cache
    # from langchain_core.caches import InMemoryCache
    ''' LANGCHAIN '''
    # from langchain.tools import Tool
    # from langchain_groq import ChatGroq
    # from langchain.chat_models import ChatOllama
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.document_loaders import WebBaseLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_core.retrievers import BaseRetriever
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # from langchain_community.vectorstores import Chroma
    from langchain_chroma import Chroma
    # from langchain.agents import load_tools
    # ''' RAG '''
    # from langchain_community.vectorstores import Chroma
    # from langchain_community.tools import DuckDuckGoSearchRun

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))


# @CrewBase
class agentWorkLoads():

    def __init__(
        self,
        desc : str="shipping schedule search and store", # identifier for the instances
        job_id:str=None # unique identifier associated with the author processes
        ) -> None:

        self.__name__ = __name__
        self.__package__ = __package__
        self.__module__ = __module__
        self.__app__ = __app__
        self.__ini_fname__ = __ini_fname__
        self.__conf_fname__ = __conf_fname__
        self.__desc__ = desc

        __s_fn_id__ = f"{self.__name__} function <__init__>"

        __def_job_id__="def_job123"

        if job_id is None or "".join(job_id.split())=="":
            self._job_id = __def_job_id__
        else:
            self._job_id = job_id

        global logger
        global pkgConf
        global appConf
        global clsLLM
        global clsVDB

        try:
            self.cwd=os.path.dirname(__file__)
            pkgConf = configparser.ConfigParser()
            pkgConf.read(os.path.join(self.cwd,__ini_fname__))

            self.projHome = pkgConf.get("CWDS","PROJECT")
            sys.path.insert(1,self.projHome)

            ''' initialize the logger '''
            from rezaware.utils import Logger as logs
            logger = logs.get_logger(
                cwd=self.projHome,
                app=self.__app__, 
                module=self.__module__,
                package=self.__package__,
                ini_file=self.__ini_fname__)
            ''' set a new logger section '''
            logger.info('########################################################')
            logger.info("%s %s",self.__name__,self.__package__)

            ''' LLM '''
            from rezaware.modules.ml.llm import model as md
            clsLLM = md.llmWorkLoads(
                    desc=self.__desc__,
                    provider="groq", #"ollama",  
                    llm_name="llama-3.3-70b-versatile", #"gemma:2b", 
                    temperature=0.2,
                    max_tokens =300,
                    max_retries=0,
                    base_url="http://192.168.2.200:5050/",
            )
            ''' get the model '''
            self.llm_model = clsLLM.get()
            if not self.llm_model:
                raise ChildProcessError("Failed to set LLM")
            logger.debug("%s LLM set with %s", __s_fn_id__, self.llm_model)

            ''' AGENTS '''
            from wrangler.modules.shipping.emailPulse import agents
            agents = agents.ScraperAgents(llm=self.llm_model)
            self.reader=agents.content_reader()
            ''' TASKS '''
            from wrangler.modules.shipping.emailPulse import tasks
            tasks = tasks.ScraperTask()
            self.read_emails=tasks.content_read_task(agent=self.reader)

            self._dbRoot = os.path.join(pkgConf.get("CWDS","DATA"),self._job_id)

            ''' VECTORDB '''
            from rezaware.modules.etl.loader import vectorDB
            clsVDB = vectorDB.dataWorkLoads(
                db_type='chromadb', 
                db_root=self._dbRoot, 
                db_name="emails")

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None

    @crew
    def crew(self) -> Crew:

        log_file = "/home/nuwan/workspace/advantis/wrangler/data/shipping/emailPulse/crew_full_output.log"
        return Crew(
            agents= [self.reader], #self.scraper,
            tasks = [self.read_emails], #self.read_txt,
            process=Process.sequential,
            # manager_llm=self.llm_model,
            output_log_file = log_file,
            full_output = True,
            verbose = True
        )

    # Function to store data in ChromaDB
    def store_in_chromadb(self,chunks, db_name, collection_name):
        # embeddings = HuggingFaceEmbeddings()
        # db_path = f"/home/nuwan/workspace/penman/wrangler/db/chroma_db_{collection_name}"
        # vectordb = Chroma.from_documents(
        #     documents,
        #     embeddings,
        #     persist_directory=db_path #f"./chroma_db_{collection_name}"
        # )
        clsVDB.dbType='chromadb'
        
        # Add documents
        # for idx, doc in enumerate(chunks):
        #     collection.add(
        #         documents=[doc],
        #         metadatas=[{"id": idx}],
        #         ids=[str(idx)]  # Unique ID for each document
        #     )
        return clsVDB.store_vectors(
            documents=chunks, # list of document chunks
            db_name = db_name,    # optional folder to append to the root
            collection=collection_name,   # the documents collection name
            embedding_fn=None, # embediing function to use
            # **kwargs,
            )
        # return vectordb

    ''' Run the crew '''
    def _run(self, inputs):

        result = self.crew().kickoff(inputs=inputs)
        chunks = clsVDB.text_to_chunks(text=result.raw, chunk_size=200, overlap=20)
        print(len(chunks))
        # vectorstore_ = self.store_in_chromadb(
        #     chunks=chunks,
        #     db_name="email",
        #     collection_name="nuwan_waidyanatha")
        # print("\nvectorstore_", vectorstore_)

        # Process task content
        # print(type(self.read_emails.output),self.read_emails.output.raw)
        # content = self.get_search_content(task=self.research_topic)
        # log_file = "/home/nuwan/workspace/advantis/wrangler/data/shipping/schedules/crew_full_output.log"
        # content = agentWorkLoads.read_crew_log(log_fpath=log_file)
        # # print(content)
        # # Split content into chunks with overlap
        # # chunks = aiWorkLoads.text_to_chunks(text=content,
        # #                                     chunk_size=200, overlap=20)
        # chunks = clsVDB.text_to_chunks(text=content, chunk_size=200, overlap=20)
        # # # Store chunks in ChromaDB
        # vectorstore_=self.store_in_chromadb(chunks,"schedule", "results")
        # print("vectorstore_", vectorstore_)

        # print("Data has been successfully stored to %s in ChromaDB." % str("results"))

        return result #, search_results

    # def get_search_content(task):
    #     """
    #     Extracts search content from a CrewAI agent task.
    #     Args:
    #         task (dict): The CrewAI agent task object.
    #     Returns:
    #         str: Consolidated search content as a single string.
    #     """
    #     # Extract content assuming `search_results` is part of the task object
    #     search_results = task.get("search_results", [])
    #     return " ".join([result.get("content", "") for result in search_results])

    # @staticmethod
    # def split_text_with_overlap(text, chunk_size=200, overlap=20):
    #     """
    #     Splits text into overlapping chunks.
    #     Args:
    #         text (str): The input text.
    #         chunk_size (int): The size of each chunk.
    #         overlap (int): The number of overlapping characters between chunks.
    #     Returns:
    #         list: List of text chunks.
    #     """

    #     chunks = []
    #     start = 0
    #     while start < len(text):
    #         end = min(start + chunk_size, len(text))
    #         chunks.append(text[start:end])
    #         start += chunk_size - overlap
    #     return chunks

    # ''' Function --- TEXT TO CHUNKS ---

    #     authors: <nuwan@soulfish.lk>
    # '''
    # @staticmethod
    # def text_to_chunks(
    #     text:list=None,
    #     chunk_size:int=1000, 
    #     overlap:int=200,
    #     **kwargs
    # )->List:
    #     """
    #     Description:
    #         Split the text to chunks
    #     Attributes :
    #         folder_path (str) directing to the folder
    #     Returns :
    #         documents (list)
    #     Exceptions :
    #         Incorrect folder path raizes exception
    #         Folder with no PDFs raises an exception
    #     """

    #     __s_fn_id__ = f"{aiWorkLoads.__name__} function <text_to_chunks>"

    #     try:
    #         ''' validate inputs '''
    #         if not isinstance(text,list) or len(text)<=0:
    #             raise AttributeError("Invalid %s text" % type(text))
    #         if not isinstance(chunk_size,int) and chunk_size<=0:
    #             raise AttributeError("Invalid chunk_size %d must be > 0; typically 1000")
    #         if not isinstance(overlap,int) and overlap<0:
    #             raise AttributeError("Invalid overlap %d must be >= 0")
    #         logger.debug("%s Splitting %d text documents into %d chunks with %d overlap", 
    #                      __s_fn_id__, len(text), chunk_size, overlap)
    #         ''' split the text '''
    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=chunk_size, 
    #             chunk_overlap=overlap
    #         )
    #         chunks = text_splitter.split_documents(text)
    #         if not isinstance(chunks,list) or len(chunks)<=0:
    #             raise RuntimeError("Failed split %d text document" % len(text))

    #     except Exception as err:
    #         logger.error("%s %s \n",__s_fn_id__, err)
    #         logger.debug(traceback.format_exc())
    #         print("[Error]"+__s_fn_id__, err)
    #         return None

    #     finally:
    #         logger.info("%s Split %d document into %d chunks", __s_fn_id__, len(text), len(chunks))
    #         return chunks

    # # Function to load and process web content
    # def load_and_process_url(self,url):
    #     loader = WebBaseLoader(url)
    #     data = loader.load()
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #     return text_splitter.split_documents(data)