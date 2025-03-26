#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import yaml
from crewai.project import task
from crewai import Task

class ScraperTask():

    def __init__(self) -> None:

        ''' TODO read yaml.safe_load through sparkFile for all storage '''
        conf_file_path = "/home/nuwan/workspace/advantis/wrangler/modules/shipping/schedules/config/tasks.yaml"
        with open(conf_file_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

        ''' TOOLS '''
        from wrangler.modules.shipping.schedules.tools import web_tools as wt
        web = wt.searchWorkLoads()
        self.web_tool = web.duckduckgo()
        from wrangler.modules.shipping.schedules.tools import api_tools as at
        api = at.searchWorkLoads()
        self.api_tool = api.scrape_freight_tool()

    @task
    def organize_content_task(self,agent) -> Task:

        _out_file="schedule.txt"
        ''' TODO output_json=outlineJSONObj, '''
        return Task(
            config = self.tasks_config['organize_content_task'],
            agent = agent, #self.article_planner(),
            async_execution=True,  # at least one must be false
            output_file = _out_file,
            # output_format='JSON'
        )

    @task
    def web_crawl_task(self,agent) -> Task:

        _out_file="web_search.txt"
        ''' TODO output_json=outlineJSONObj, '''
        return Task(
            config = self.tasks_config['web_crawl_task'],
            agent = agent, #self.article_planner(),
            async_execution=False,  # at least one must be false
            output_file = _out_file,
            # output_format='JSON'
        )

    @task
    def scrape_content_task(self,agent) -> Task:
        
        _out_file="url_search.txt"
        return Task(
            config = self.tasks_config['scrape_content_task'],
            agent = agent, #self.topics_researcher(),
            async_execution=False,
            tools=[self.api_tool],
            output_file = _out_file,
            # allow_delegation=True,
        )