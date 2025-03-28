#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import yaml
from crewai.project import task
from crewai import Task

class ScraperTask():

    def __init__(self) -> None:

        ''' TODO read yaml.safe_load through sparkFile for all storage '''
        conf_file_path = "/home/nuwan/workspace/advantis/wrangler/modules/shipping/emailPulse/config/tasks.yaml"
        with open(conf_file_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

        ''' TOOLS '''
        from wrangler.modules.shipping.emailPulse.tools import read_tool as rt
        clsRead = rt.searchWorkLoads()
        self.email_tool = clsRead.read_email_tool()

    @task
    def content_read_task(self,agent) -> Task:
        
        _out_file="emails.txt"
        return Task(
            config = self.tasks_config['content_read_task'],
            agent = agent, #self.topics_researcher(),
            async_execution=False,
            tools=[self.email_tool],
            output_file = _out_file,
            # allow_delegation=True,
        )