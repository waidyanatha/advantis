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

        conf_file_path = "/home/nuwan/workspace/advantis/wrangler/modules/shipping/emailPulse/config/agents.yaml"
        with open(conf_file_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)

        ''' TOOLS '''
        from wrangler.modules.shipping.emailPulse.tools import read_tool as rt
        clsRead = rt.searchWorkLoads()
        self.email_tool = clsRead.read_email_tool()

    @agent
    def content_reader(self) -> Agent:
        return Agent(
            config = self.agents_config['content_reader'],
            llm = self.llm,
            tools=[self.email_tool],
            allow_delegation=False,
            cache=True,
        )
