#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import yaml
from crewai.project import task
from crewai import Task

class ScraperTask():

    def __init__(self) -> None:

        ''' TODO read yaml.safe_load through sparkFile for all storage '''
        conf_file_path = "/home/nuwan/workspace/advantis/mining/modules/shipping/emailPulse/config/tasks.yaml"
        with open(conf_file_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

        ''' TOOLS '''
        from mining.modules.shipping.emailPulse.tools import rag
        clsRAG = rag.searchWorkLoads()
        self.analyst_tool = clsRAG.query_email_tool()

    @task
    def content_analysis_task(self,agent) -> Task:

        _out_file="analys_report.txt"
        return Task(
            config = self.tasks_config['content_analysis_task'],
            agent = agent, #self.topics_researcher(),
            async_execution=False,
            tools=[self.analyst_tool],
            # input={"query": "Analyze all emails related to project deadlines."},
            output_file = _out_file,
            # allow_delegation=True,
        )