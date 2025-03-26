#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import yaml
from crewai.project import agent
from crewai import Agent
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool

class ScraperAgents():

    def __init__(self, llm) -> None:

        self.llm = llm
#         self.search_tool = DuckDuckGoSearchRun()

        conf_file_path = "/home/nuwan/workspace/advantis/wrangler/modules/shipping/schedules/config/agents.yaml"
        with open(conf_file_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)

        ''' TOOLS '''
        from wrangler.modules.shipping.schedules.tools import web_tools as wt
        web = wt.searchWorkLoads()
        self.web_tool = web.duckduckgo()
        from wrangler.modules.shipping.schedules.tools import api_tools as at
        api = at.searchWorkLoads()
        self.api_tool = api.scrape_freight_tool()


    @agent
    def content_manager(self) -> Agent:
        return Agent(
            config = self.agents_config['content_manager'],
            llm = self.llm,
            allow_delegation=True,
            cache=True,
        )

    @agent
    def web_researcher(self) -> Agent:

        return Agent(
            config = self.agents_config['web_researcher'],
            llm = self.llm,
            allow_delegation=False,
            tools=[self.web_tool], # tools=[self.search_tool],
            cache=True,
        )

    @agent
    def url_scraper(self) -> Agent:

        return Agent(
            config = self.agents_config['url_scraper'],
            llm = self.llm,
            allow_delegation=False,
            tools=[self.api_tool], # tools=[self.search_tool],
            cache=True,
        )