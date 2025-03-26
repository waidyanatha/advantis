#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool

class searchWorkLoads():

    def __init__(self) -> None:
        return None

    def scrape_sea_freight_details(url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
    
            soup = BeautifulSoup(response.text, 'html.parser')
    
            # Example: Look for text blocks that may contain freight details
            content = soup.get_text(separator='\n')
    
            # Simple heuristic extraction (can be customized)
            lines = content.split('\n')
            extracted_lines = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ["origin", "destination", "carrier", 
                                                               "port", "freight", "transit", "frequency"]):
                    extracted_lines.append(line.strip())
    
            if not extracted_lines:
                return "No relevant sea freight details found."
    
            return "\n".join(extracted_lines)
    
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"
    
    # CrewAI-compatible tool definition
    # scrape_freight_tool = Tool(
    #     name="SeaFreightScraper",
    #     description="Scrapes a provided URL to retrieve origin-to-destination sea freight details, port info, and transit times.",
    #     func=scrape_sea_freight_details
    # )
    def scrape_freight_tool(self)-> Tool:

        # return Tool(
        #     name="DuckDuckGo Search",
        #     func=search.run,
        #     description="Useful for searching the internet for recent information."
        # )
        return Tool(
            name="SeaFreightScraper",
            description="Scrapes a URL to retrieve origin-to-destination sea freight details.",
            func=searchWorkLoads.scrape_sea_freight_details
            )
