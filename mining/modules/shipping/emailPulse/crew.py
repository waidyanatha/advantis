#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

''' Initialize with default environment variables '''
__name__ = "crew"
__package__ = "emailPulse"
__module__ = "shipping"
__app__ = "mining"
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
    from datetime import datetime
    # from dotenv import load_dotenv
    # load_dotenv()

    # from typing import List, Iterable, Dict, Tuple
    # import litellm
    ''' CREWAI '''
    from crewai import Crew, Process #BaseAgent
    from crewai.project import CrewBase, crew
    # ''' caching '''
    # from langchain_core.globals import set_llm_cache
    # from langchain_core.caches import InMemoryCache
    ''' LANGCHAIN '''
    from langchain.embeddings.openai import OpenAIEmbeddings

    print("All functional %s-libraries in %s-package of %s-module imported successfully!"
          % (__name__.upper(),__package__.upper(),__module__.upper()))

except Exception as e:
    print("Some packages in {0} module {1} package for {2} function didn't load\n{3}"\
          .format(__module__.upper(),__package__.upper(),__name__.upper(),e))


# @CrewBase
class agentWorkLoads():

    def __init__(
        self,
        desc : str="shipping email analyzer", # identifier for the instances
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

        # __def_job_id__="def_job123"

        # if job_id is None or "".join(job_id.split())=="":
        #     self._job_id = __def_job_id__
        # else:
        #     self._job_id = job_id

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
            from mining.modules.shipping.emailPulse import agents
            agents = agents.ScraperAgents(llm=self.llm_model)
            self.analyst=agents.email_analyst()
            ''' TASKS '''
            from mining.modules.shipping.emailPulse import tasks
            tasks = tasks.ScraperTask()
            self.report=tasks.content_analysis_task(agent=self.analyst)

        except Exception as err:
            logger.error("%s %s \n",__s_fn_id__, err)
            logger.debug(traceback.format_exc())
            print("[Error]"+__s_fn_id__, err)

        return None

    @crew
    def crew(self) -> Crew:

        log_file = "/home/nuwan/workspace/advantis/mining/data/shipping/emailPulse/crew_full_output.log"
        return Crew(
            agents= [self.analyst], #self.scraper,
            tasks = [self.report], #self.read_txt,
            process=Process.sequential,
            # manager_llm=self.llm_model,
            output_log_file = log_file,
            full_output = True,
            verbose = True
        )

    ''' Run the crew '''
    def _run(self, inputs):

        result = self.crew().kickoff(inputs=inputs)
        return result