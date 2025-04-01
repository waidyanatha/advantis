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
    from langchain_chroma import Chroma

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
        # global clsLLM
        # global clsVDB

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
            self.clsLLM = md.llmWorkLoads(
                    desc=self.__desc__,
                    provider="groq", #"ollama",  
                    llm_name="llama-3.3-70b-versatile", #"gemma:2b", 
                    temperature=0.2,
                    max_tokens =300,
                    max_retries=0,
                    base_url="http://192.168.2.200:5050/",
            )
            ''' get the model '''
            self.llm_model = self.clsLLM.get()
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

            # self._dbRoot = os.path.join(pkgConf.get("CWDS","DATA"),self._job_id)

            ''' VECTORDB '''
            _db_type = 'chromadb'
            _db_root = os.path.join(pkgConf.get("CWDS","DATA"),self._job_id)
            _db_name = 'email'
            from rezaware.modules.etl.loader import vectorDB
            self.clsVDB = vectorDB.dataWorkLoads(
                db_type=_db_type, 
                db_root=_db_root,
                db_name=_db_name
            )

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None

    @crew
    def crew(self) -> Crew:

        # log_file = "/home/nuwan/workspace/advantis/wrangler/data/shipping/emailPulse/crew_full_output.log"
        log_file = os.path.join(pkgConf.get("CWDS","DATA"),"crew_full_output.log")
        return Crew(
            agents= [self.reader], #self.scraper,
            tasks = [self.read_emails], #self.read_txt,
            process=Process.sequential,
            # manager_llm=self.llm_model,
            output_log_file = log_file,
            full_output = True,
            verbose = True
        )

    ''' Run the crew '''
    def _run(self, inputs):

        crew_output = self.crew().kickoff(inputs=inputs)
        # Process content and store vectors in chroma
        kwargs = {"SPLITTER" : "LANGCHAIN"}
        documents = self.clsVDB.text_to_documents(
            text=crew_output.raw, 
            chunk_size=200, 
            overlap=10,
            **kwargs,
        )
        collection_name='nuwan'
        vect_store_ids_, vectorstore_ = self.clsVDB.store_vectors(
            documents=documents, # list of document documents
            collection=collection_name,   # the documents collection name
            embedding_fn=None, # embediing function to use
            )        
        # # _vectorstore_=self.store_in_chromadb(documents, collection_name)
        # print("\nvectorstore_", vectorstore_)

        # print("Data has been successfully stored to %s in ChromaDB." % str("results"))

        return crew_output, vect_store_ids_, vectorstore_ #, vectorstore_ #, search_results

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

