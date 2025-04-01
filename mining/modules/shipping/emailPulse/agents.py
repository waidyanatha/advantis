#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import yaml
from crewai.project import agent
from crewai import Agent
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool

class ScraperAgents():

    def __init__(self, llm) -> None:

        self.llm = llm
#         self.search_tool = DuckDuckGoSearchRun()

        conf_file_path = "/home/nuwan/workspace/advantis/mining/modules/shipping/emailPulse/config/agents.yaml"
        with open(conf_file_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)

        ''' TOOLS '''
        from mining.modules.shipping.emailPulse.tools import rag
        clsRAG = rag.searchWorkLoads()
        self.analyst_tool = clsRAG.query_email_tool()

    @agent
    def email_analyst(self) -> Agent:
        return Agent(
            config = self.agents_config['email_analyst'],
            llm = self.llm,
            tools=[self.analyst_tool],
            allow_delegation=False,
            cache=True,
        )