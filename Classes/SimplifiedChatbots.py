import json
import os
from datetime import datetime
import logging
import uuid
import openai 
from openai import OpenAI
import time
from google.cloud import aiplatform
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from google.api_core.exceptions import ResourceExhausted as RateLimitError
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

# New classes to use more llm

@dataclass
class Message:
    role: str
    content: str
    timestamp: str

class ModelConfig:
    """Configuration class for different model providers"""
    
    OPENAI_MODELS = {
        "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview",
        "gpt-4o-mini", "chatgpt-4o-latest", "o3-mini-2025-01-31", "o3-mini"
    }
    
    GEMINI_MODELS = {
        "gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-2.0-flash-exp", "models/gemini-1.5-pro", "gemini-1.5-pro-latest"
    }
    
    @staticmethod
    def is_openai_model(model_name: str) -> bool:
        return model_name in ModelConfig.OPENAI_MODELS
    
    @staticmethod
    def is_gemini_model(model_name: str) -> bool:
        return model_name in ModelConfig.GEMINI_MODELS

class ChatClient:
    """Abstract base class for chat clients"""
    
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

class OpenAIChatClient(ChatClient):
    def __init__(self, model_name: str, max_tokens: int, temperature: float):
        from openai import OpenAI
        self.client = OpenAI()  
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        # For models that require max_completion_tokens and do not support temperature.
        if self.model_name in {"o3-mini", "o3-mini-2025-01-31"}:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=self.max_tokens  # Use alternative parameter name.
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        return response.choices[0].message.content


class GeminiChatClient(ChatClient):
    def __init__(self, model_name: str, max_tokens: int, temperature: float):
        # Initialize Gemini client
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        # Convert messages to Gemini format
        chat = self.model.start_chat()
        
        # Process each message
        for message in messages:
            if message["role"] == "system":
                # Add system message as user message with special prefix
                chat.send_message(f"[System Instructions]: {message['content']}")
            elif message["role"] == "user":
                chat.send_message(message["content"])
            elif message["role"] == "assistant":
                # Skip assistant messages as they're responses
                continue
        
        # Generate response with configured parameters
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature
            )
        )
        
        return response.text

# Base class for common chatbot functionalities
class Chatbot:
    def __init__(self, 
                 json_save_path=None, 
                 docs_paths=None, 
                 model_name="gpt-4o-mini", 
                 max_tokens=750, 
                 temperature=0):
        """
        Initializes the chatbot
        """

        self.json_save_path = json_save_path
        self.initialize_chat_dict(json_save_path, docs_paths, model_name, max_tokens, temperature)
        
        # New: Initialize appropriate client based on model name
        if ModelConfig.is_openai_model(model_name):
            self.chat_client = OpenAIChatClient(model_name, max_tokens, temperature)
        elif ModelConfig.is_gemini_model(model_name):
            self.chat_client = GeminiChatClient(model_name, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def initialize_chat_dict(self, json_save_path, docs_paths, model_name, max_tokens, temperature):
        """New: Separated initialization logic for better organization"""
        if json_save_path:
            try:
                with open(json_save_path, 'r') as file:
                    self.chat_dict = json.load(file)
                self.docs_paths = self.chat_dict["docs_paths"]
                self.model_name = self.chat_dict["model_name"]
                self.max_tokens = self.chat_dict["max_tokens"]
                self.temperature = self.chat_dict["temperature"]
            except Exception as e:
                print(f"Error loading chat from {json_save_path}: {e}")
        else:
            try:
                self.chat_dict = {
                    "id": datetime.now().strftime("%d:%m:%y_%H.%M.%S"),
                    "docs_paths": docs_paths,
                    "model_name": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [],
                }
                self.docs_paths = docs_paths
                self.model_name = model_name
                self.max_tokens = max_tokens
                self.temperature = temperature
            except Exception as e:
                print(f"Error initializing chat: {e}")
        
        if self.json_save_path:
            try: 
                with open(self.json_save_path, 'r') as file:
                    self.chat_dict = json.load(file)

                self.docs_paths = self.chat_dict["docs_paths"]
                self.model_name = self.chat_dict["model_name"]
                self.max_tokens = self.chat_dict["max_tokens"]
                self.temperature = self.chat_dict["temperature"]
            except Exception as e:
                print(f"Error loading chat from {self.json_save_path}: {e}")
        else:
            try:
                self.chat_dict = {
                    "id": datetime.now().strftime("%d:%m:%y_%H.%M.%S"),
                    "docs_paths": docs_paths,
                    "model_name": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [],
                }

                self.docs_paths = docs_paths
                self.model_name = model_name
                self.max_tokens = max_tokens
                self.temperature = temperature
            except Exception as e:
                print(f"Error initializing chat: {e}")

    def add_message(self, message, role="user"):
        """
        Adds a message to the chat dictionary.
        """
        self.chat_dict["messages"].append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

    def ask(self, question):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_chat_to_json(self, folder_path):
        """
        Saves the chat to a JSON file.
        """
        filename = f"chat_{self.chat_dict['id']}.json"
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'w') as file:
                json.dump(self.chat_dict, file, indent=4)
        except Exception as e:
            print(f"Error saving chat to {file_path}: {e}")

    def save_chat_to_md(self, folder_path):
        """
        Saves the chat to a markdown file.
        """
        filename = f"chat_{self.chat_dict['id']}.md"
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'w') as file:
                for message in self.chat_dict["messages"]:
                    file.write(f"## {message['role'].capitalize()}\n\n{message['content']}\n\n")
        except Exception as e:
            print(f"Error saving chat to {file_path}: {e}")

class ChatGPT(Chatbot):
    def __init__(self, 
                 system_prompt="You are a helpful assistant!",
                 max_retries=3,
                 retry_delay=2,
                 **kwargs):
        """
        Initializes the GPT chatbot.
        """
        super().__init__(**kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        try:
            # For models like "o3-mini" that don't support temperature,
            # use max_completion_tokens and omit temperature.
            if self.model_name in {"o3-mini", "o3-mini-2025-01-31"}:
                self.chat_openAI = ChatOpenAI(
                    model_name=self.model_name,
                    max_completion_tokens=self.max_tokens,
                    request_timeout=90  # Increased timeout
                )
            else:
                self.chat_openAI = ChatOpenAI(
                    model_name=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    request_timeout=90,  # Increased timeout
                )
            self.add_message(system_prompt, role="system")
        except Exception as e:
            print(f"Error initializing GPT chatbot: {e}")

    def ask(self, question):
        """
        Asks the chatbot a question with retry logic for rate-limit (429) errors.
        """
        self.add_message(question, role="user")
        attempt = 0
        delay = self.retry_delay
        while attempt < self.max_retries:
            try:
                # Using the abstracted chat_client
                ai_response = self.chat_client.invoke(self.chat_dict["messages"])
                if ai_response:
                    self.add_message(ai_response, role="assistant")
                    # Optionally save chat history
                    try:
                        self.save_chat_to_json("Chats_Database/JSON")
                        self.save_chat_to_md("Chats_Database/MD")
                    except Exception as save_error:
                        print(f"Warning: Could not save chat history: {save_error}")
                    return ai_response
            except RateLimitError as e:
                print(f"Rate limit reached on attempt {attempt + 1}: {e}. Waiting {delay} seconds.")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                attempt += 1
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(delay)
                delay *= 2
                attempt += 1

        raise Exception(f"Failed after {self.max_retries} attempts due to rate limit or other errors.")

class BigSummarizerGPT(Chatbot):
    def __init__(self, 
                 system_prompt="You are a helpful assistant!",
                 **kwargs):
        super().__init__(**kwargs)
        self.add_message(system_prompt, role="system")

    def get_document_loader(self, document_path: str) -> str:
        """
        Loads the content of a document based on its file type.
        
        Args:
            document_path (str): Path to the document
            
        Returns:
            str: The content of the document as a string.
        """
        document_path = document_path.strip()
        
        if document_path.endswith('.pdf'):
            try:
                reader = PdfReader(document_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text.strip()
            except Exception as e:
                raise ValueError(f"Failed to load PDF file: {str(e)}")
        
        elif document_path.endswith(('.txt', '.md')):
            try:
                with open(document_path, 'r', encoding='utf-8') as file:
                    return file.read().strip()
            except Exception as e:
                raise ValueError(f"Failed to load text/markdown file: {str(e)}")
        
        else:
            raise ValueError(f"Unsupported file type for document: {document_path}")

    def split_text(self, text, max_chunk_size=20500):
        chunks = []
        current_chunk = ""
        for sentence in text.split("."):
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + "."
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def crop_dict(self, chat_bot_dict: Dict, max_token: int) -> Dict:
        """
        Trims the chat history to ensure the total token count stays within the limit.
        
        Args:
            chat_bot_dict (dict): The chat dictionary containing messages.
            max_token (int): The maximum allowable token count.
        
        Returns:
            dict: The trimmed chat dictionary.
        """
        total_character = 0
        chat_bot_dict["messages"].reverse()  # Start removing from the oldest messages
        for message in chat_bot_dict["messages"][:-1]:  # Exclude the last message (current query)
            total_character += len(message["content"])
            # Divide by 4 to estimate tokens from characters (approximation)
            if total_character / 4 > max_token:
                chat_bot_dict["messages"].remove(message)
        chat_bot_dict["messages"].reverse()  # Restore original order
        return chat_bot_dict

    def ask(self, question, document_path, max_chunk_size=20000, dict_max_token=15000, max_retries=3, retry_delay=2):
        """
        Asks the chatbot a question about a document.

        Args:
            question (str): The question to ask.
            document_path (str): Path to the document.
            max_chunk_size (int): Maximum size of text chunks.
            dict_max_token (int): Maximum number of tokens in the chat history.
            max_retries (int): Maximum number of retries per chunk.
            retry_delay (int): Initial delay (in seconds) before retrying.

        Returns:
            str: Concatenated summaries from all chunks.
        """
        try:
            # Validate the document path
            if not document_path or not isinstance(document_path, str):
                raise ValueError("Invalid document path provided")

            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document not found at path: {document_path}")

            # Load document content
            doc_content = self.get_document_loader(document_path)
            if not doc_content.strip():
                raise ValueError("Document appears to be empty")

            chunks = self.split_text(doc_content, max_chunk_size)
            previous_summaries = []

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk: {i+1}/{len(chunks)}")

                system_prompt = f"""You are a helpful assistant!
Previous summaries: {' '.join(previous_summaries)}"""

                new_bot = BigSummarizerGPT(
                    system_prompt=system_prompt,
                    model_name=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )

                user_prompt = f"""{question}\n\nHere is the document section to analyze:\n\n{chunk}"""
                new_bot.add_message(user_prompt, role="user")
                new_bot.crop_dict(new_bot.chat_dict, dict_max_token)

                # Retry loop for this chunk
                attempt = 0
                delay = retry_delay
                response = None
                while attempt < max_retries:
                    try:
                        response = new_bot.chat_client.invoke(new_bot.chat_dict["messages"])
                        break  # Success: exit the retry loop
                    except RateLimitError as e:
                        print(f"Attempt {attempt + 1} for chunk {i+1} failed with rate-limit error: {e}. Waiting {delay} seconds.")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                        attempt += 1
                    except Exception as e:
                        print(f"Attempt {attempt + 1} for chunk {i+1} failed with error: {e}. Waiting {delay} seconds.")
                        time.sleep(delay)
                        delay *= 2
                        attempt += 1

                if response is None:
                    print(f"Skipping chunk {i+1} after {max_retries} attempts due to errors.")
                    continue  # Skip this chunk if still failing
                
                new_bot.add_message(response, role="assistant")
                previous_summaries.append(response)

                # Save chat history
                try:
                    new_bot.save_chat_to_json("Chats_Database/JSON")
                    new_bot.save_chat_to_md("Chats_Database/MD")
                except Exception as save_error:
                    print(f"Warning: Could not save chat history: {save_error}")

            return "\n\n".join(previous_summaries)


        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise
