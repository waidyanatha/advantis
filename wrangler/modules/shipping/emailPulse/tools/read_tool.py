#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import base64
import pickle
from datetime import datetime
from langchain.tools import Tool
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

class searchWorkLoads:
    def __init__(self) -> None:
        super().__init__()
        self.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        self.service = self.authenticate_gmail()

    def authenticate_gmail(self):
        """Authenticate with Gmail API"""
        creds = None
        token_path = "token.pickle"
        credentials_path = "/home/nuwan/workspace/advantis/wrangler/modules/shipping/emailPulse/tools/credentials.json"

        if os.path.exists(token_path):
            with open(token_path, "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, self.scopes)
                creds = flow.run_local_server(port=0)
                with open(token_path, "wb") as token:
                    pickle.dump(creds, token)

        return build("gmail", "v1", credentials=creds)

    def get_label_ids(self):
        """Retrieves and prints all Gmail label IDs."""
        labels_response = self.service.users().labels().list(userId="me").execute()
        return labels_response.get("labels", [])

    def fetch_emails(self, from_date=None, to_date=None, max_results=4):
        """Fetch recent emails with optional filtering."""
        
        query = []
        if from_date:
            query.append(f"after:{from_date}")
        if to_date:
            query.append(f"before:{to_date}")
        
        query_str = " ".join(query) if query else None
    
        results = self.service.users().messages().list(
            userId="me", labelIds=["CATEGORY_PERSONAL"], q=query_str, maxResults=max_results
        ).execute()
        
        messages = results.get("messages", [])
        if not messages:
            print("No new emails matching the filters.")
            return []
        
        emails = []
        for msg in messages:
            msg_data = self.service.users().messages().get(userId="me", id=msg["id"]).execute()
            payload = msg_data.get("payload", {})
            headers = payload.get("headers", [])
    
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
    
            parts = payload.get("parts", [])
            email_body = ""
    
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    email_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
    
            emails.append({"subject": subject, "from": sender, "body": email_body})
    
        return emails

    def read_email_tool(self) -> Tool:
        """Creates a LangChain Tool to fetch emails"""
        # return Tool(
        #     name="Email Reader",
        #     description="Fetches recent emails from the user's inbox based on filters.",
        #     func=lambda inputs: self.fetch_emails(
        #         from_date=inputs.get("from_date"),
        #         to_date=inputs.get("to_date"),
        #     ),
        # )
        return Tool(
            name="Email Reader",
            description="Fetches recent emails from the user's inbox based on filters.",
            func=lambda inputs: self.fetch_emails(
                from_date=inputs["from_date"] if isinstance(inputs, dict) and "from_date" in inputs else None,
                to_date=inputs["to_date"] if isinstance(inputs, dict) and "to_date" in inputs else None,
            ),
        )