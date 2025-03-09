import json
import os
from datetime import datetime
import logging
import uuid
import openai 
from openai import OpenAI
import time
from google.cloud import aiplatform
from vertexai.language_models import ChatModel, TextGenerationModel
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_core.documents.base import Document

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
        "gpt-4o-mini", "chatgpt-4o-latest"
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
                 model_name="gpt-3.5-turbo", 
                 max_tokens=750, 
                 temperature=0.1):
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

class CompletionBot(Chatbot):
    def __init__(self, 
                 **kwargs):
        """
        Initializes the completion chatbot.
        """
        super().__init__(**kwargs)

        try:
            self.chat_openAI = OpenAI()

        except Exception as e:
            print(f"Error initializing GPT chatbot: {e}")

    def ask(self, system_prompt):
        """
        Asks the chatbot a question.
        """
        self.add_message(system_prompt, role="system")

    
        response = self.chat_openAI.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            prompt=system_prompt,
            )
        
        self.add_message(response.choices[0].text, role="response")


        self.save_chat_to_json("Chats_Database/JSON")
        self.save_chat_to_md("Chats_Database/MD")

        return response.choices[0].text
        #return response

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
            self.chat_openAI = ChatOpenAI(model_name=self.model_name,
                                        max_tokens=self.max_tokens,
                                        temperature=self.temperature,
                                        request_timeout=90,  # Increased timeout
                                        )
            self.add_message(system_prompt, role="system")
        except Exception as e:
            print(f"Error initializing GPT chatbot: {e}")

    def ask(self, question):
        """
        Asks the chatbot a question with retry logic.
        """
        self.add_message(question, role="user")
        
        for attempt in range(self.max_retries):
            try:
                # Using the abstracted chat_client instead of chat_openAI
                ai_response = self.chat_client.invoke(self.chat_dict["messages"])
                
                if ai_response:  # Only proceed if we got a valid response
                    self.add_message(ai_response, role="assistant")
                    
                    # Save chat history
                    try:
                        self.save_chat_to_json("Chats_Database/JSON")
                        self.save_chat_to_md("Chats_Database/MD")
                    except Exception as save_error:
                        print(f"Warning: Could not save chat history: {save_error}")

                    return ai_response
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")

        raise Exception("Failed to get valid response after all retry attempts")
    
class SummarizerGPT(Chatbot):
    def __init__(self, 
                 system_prompt="You are a summarizer assistant!",
                 **kwargs):
        """
        Initializes the GPT chatbot.
        """
        super().__init__(**kwargs)

        try:
            self.chat_openAI = ChatOpenAI(model_name=self.model_name,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          # top_p=0.9,
                                          )
            self.add_message(system_prompt, role="system")
        except Exception as e:
            print(f"Error initializing GPT chatbot: {e}")

    # Changed: Made document parameter optional by adding `document=None`
    def ask(self, question, document=None):
        """
        Asks the chatbot a question, optionally attaching a document.
        """
        # Added: Initialize `doc_content` as an empty string in case no document is provided
        doc_content = "" 
        
        # Conditional block to process the document only if it is provided
        if document:
            if isinstance(document, Document):
                doc_content = document.page_content
            else:
                try:
                    if document.endswith('.pdf') or document.endswith('.PDF'):
                        loader = PyPDFLoader(document)
                        document = loader.load()
                        for doc in document:
                            doc_content += doc.page_content
                    elif document.startswith('http'):
                        loader = WebBaseLoader(document)
                        document = loader.load()
                        for doc in document:
                            doc_content += doc.page_content
                    else:
                        loader = TextLoader(document)
                        document = loader.load()
                        for doc in document:
                            doc_content += doc.page_content
                except Exception as e:
                    print(f"Error loading document from {document}: {e}")

        user_prompt = f"{question}\n" + doc_content
        self.add_message(user_prompt, role="user")
        
        try:
            # Using the abstracted chat_client instead of chat_openAI
            ai_response = self.chat_client.invoke(self.chat_dict["messages"])
            self.add_message(ai_response, role="assistant")
        except Exception as e:
            print(f"Error invoking chatbot: {e}")

        self.save_chat_to_json("Chats_Database/JSON")
        self.save_chat_to_md("Chats_Database/MD")

        return ai_response
    
class BigSummarizerGPT(Chatbot):
    def __init__(self, 
                 system_prompt="You are a helpful assistant!",
                 **kwargs):
        super().__init__(**kwargs)
        self.add_message(system_prompt, role="system")

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

    def crop_dict(self, chat_bot_dict, max_token):
        total_character = 0
        chat_bot_dict["messages"].reverse()
        for message in chat_bot_dict["messages"][:-1]:
            total_character += len(message["content"])

            if total_character/4 > max_token:
                chat_bot_dict["messages"].remove(message)

        chat_bot_dict["messages"].reverse()
        return chat_bot_dict

    def get_document_loader(self, document_path):
        """
        Returns the appropriate document loader based on file extension or URL.
        Raises ValueError if the file type is not supported.
        """
        document_path = document_path.strip()
        
        if document_path.endswith('.pdf'):
            try:
                return PyPDFLoader(document_path)
            except Exception as e:
                raise ValueError(f"Failed to load PDF file: {str(e)}")
        elif document_path.startswith(('http://', 'https://')):
            return WebBaseLoader(document_path)
        elif document_path.endswith(('.txt', '.md')):
            return TextLoader(document_path)
        else:
            raise ValueError(f"Unsupported file type for document: {document_path}")

    def ask(self, question, document_path, max_chunk_size=20000, dict_max_token=15000):
        """
        Asks the chatbot a question about a document.
        
        Args:
            question (str): The question to ask
            document_path (str): Path to the document
            max_chunk_size (int): Maximum size of text chunks
            dict_max_token (int): Maximum number of tokens in the chat history
            
        Returns:
            str: Concatenated summaries from all chunks
        """
        try:
            # Validate inputs
            if not document_path or not isinstance(document_path, str):
                raise ValueError("Invalid document path provided")
            
            if not os.path.exists(document_path) and not document_path.startswith(('http://', 'https://')):
                raise FileNotFoundError(f"Document not found at path: {document_path}")

            # Get appropriate loader and load document
            loader = self.get_document_loader(document_path)
            try:
                documents = loader.load()
            except Exception as e:
                raise ValueError(f"Error loading document: {str(e)}")

            if not documents:
                raise ValueError("No content loaded from document")

            # Concatenate document content
            doc_content = "\n".join(doc.page_content for doc in documents)
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

                try:
                    # Use chat_client instead of chat_openAI
                    response = new_bot.chat_client.invoke(new_bot.chat_dict["messages"])
                    new_bot.add_message(response, role="assistant")
                    previous_summaries.append(response)
                except Exception as e:
                    print(f"Error in chunk {i+1} processing: {str(e)}")
                    continue

                # Save chat history
                new_bot.save_chat_to_json("Chats_Database/JSON")
                new_bot.save_chat_to_md("Chats_Database/MD")

            return " ".join(previous_summaries)

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise

class GPTs(Chatbot):
    def __init__(self,
                 assistant_name="Math Tutor",
                 instructions="Your name is henri, You are a helpful assistant!", 
                 tools=[{"type": "code_interpreter"}],
                 assistant_id=None,
                 **kwargs):
        """
        Initializes the GPTs chatbot.
        """
        super().__init__(**kwargs)

        try:
            self.client = openai.OpenAI()
        except Exception as e:
            print(f"Error initializing GPTs chatbot: {e}")

        if self.json_save_path:
            with open(self.json_save_path, 'r') as file:
                self.chat_dict = json.load(file)

            self.assistant_name = self.chat_dict["assistant_name"]
            self.instructions = self.chat_dict["instructions"]
            self.tools = self.chat_dict["tools"]
            self.assistant_id = self.chat_dict["assistant_id"]
            self.thread_id = self.chat_dict["thread_id"]
        else:

            if assistant_id is None:
                assistant = self.client.beta.assistants.create(
                    name=assistant_name,
                    instructions=instructions,
                    tools=tools,
                    model=self.model_name,
                )

                assistant_id = assistant.id
            
            # Create a Thread
            thread = self.client.beta.threads.create()

            self.chat_dict["assistant_name"] = assistant_name
            self.chat_dict["instructions"] = instructions
            self.chat_dict["tools"] = tools
            self.chat_dict["assistant_id"] = assistant_id
            self.chat_dict["thread_id"] = thread.id

            self.assistant_name = assistant_name
            self.instructions = instructions
            self.tools = tools
            self.assistant_id = assistant_id
            self.thread_id = thread.id


    def ask(self, question, files_path=None):
        """
        Asks the chatbot a question.
        """

        if files_path:
            files = []
            for file_path in files_path:
                try:
                    file = self.client.files.create(
                        file=open(file_path, 'rb'),
                        purpose='assistants'
                        )
                    file.filename = file_path.split("/")[-1]
                    files.append(file)
                except Exception as e:
                    print(f"Error uploading file: {e}")

        self.add_message(question, role="user")

        if files_path:
            print("Uploading files")
            response = self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=question,
                file_ids=[f.id for f in files]
            )
        else:
            self.client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=question
            )

        
        # Create a Run
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
                #instructions="Please address the user directly. The user has a premium account."
            )  
        except Exception as e:
            print(f"Error creating run: {e}")
        
        # Polling for the Run to complete
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)  # Wait for 1 second before polling again
            print(f"Run status: {run.status}")
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
        
        # Once the Run completes, retrieve and print the Messages
        if run.status == 'completed': 
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread_id
            )
            
            # Assuming you want to return the last message from the assistant
            # Adjust the following code to fit your use case
            if messages.data:
                if files_path:
                    for i, file in enumerate(files):
                        messages.data[0].content[0].text.value = messages.data[0].content[0].text.value.replace(file.id, file.filename)
                    messages.data[0].content[0].text.value = messages.data[0].content[0].text.value.replace("/".join(files_path[0].split("/")[:-1]) + "/", "")
                self.add_message(messages.data[0].content[0].text.value, role="assistant")
                self.save_chat_to_json("Chats_Database/JSON")
                self.save_chat_to_md("Chats_Database/MD")
                return messages.data[0].content[0].text.value
            else:
                return "The assistant did not provide a response."
        else:
            return f"Run ended with status: {run.status}"



# New base class for chatbots with retrieval functionality
class RetrieverChatbot(Chatbot):
    def __init__(self, 
                 system_prompt="You are a helpful assistant!", 
                 preprompt="Here are some relevant documents that might be helpful for the question:\n",
                 question_prompt="Question: ",
                 retriever=None,
                 embedding_model="text-embedding-3-small", 
                 splitter_chunk_size=500, 
                 splitter_chunk_overlap=50, 
                 k_docs=25,
                 **kwargs):
        """
        Initializes the retriever chatbot.
        """
        super().__init__(**kwargs)

        try:
            self.chat_openAI = ChatOpenAI(model_name=self.model_name, 
                                        max_tokens=self.max_tokens, 
                                        temperature=self.temperature)
        except Exception as e:
            print(f"Error initializing GPT chatbot: {e}")
        
        if self.json_save_path:
            self.system_prompt = self.chat_dict["system_prompt"]
            self.preprompt = self.chat_dict["preprompt"]
            self.question_prompt = self.chat_dict["question_prompt"]
            self.embedding_model = self.chat_dict["embedding_model"]
            self.splitter_chunk_size = self.chat_dict["splitter_chunk_size"]
            self.splitter_chunk_overlap = self.chat_dict["splitter_chunk_overlap"]
            self.k_docs = self.chat_dict["k_docs"]
        else:
            self.system_prompt = system_prompt
            self.preprompt = preprompt
            self.question_prompt = question_prompt
            self.embedding_model = embedding_model
            self.splitter_chunk_size = splitter_chunk_size
            self.splitter_chunk_overlap = splitter_chunk_overlap
            self.k_docs = k_docs

            self.chat_dict["system_prompt"] = system_prompt
            self.chat_dict["preprompt"] = preprompt
            self.chat_dict["question_prompt"] = question_prompt
            self.chat_dict["embedding_model"] = embedding_model
            self.chat_dict["splitter_chunk_size"] = splitter_chunk_size
            self.chat_dict["splitter_chunk_overlap"] = splitter_chunk_overlap
            self.chat_dict["k_docs"] = k_docs

        if self.json_save_path:
            self.add_message(self.system_prompt, role="system")

        self.retriever = retriever

    def get_retriever(self):
        """
        Returns the document retriever.
        """
        return self.retriever
        
    def set_preprompt(self, preprompt):
        """
        Sets the preprompt for the chatbot.
        """
        self.preprompt = preprompt

    def set_question_prompt(self, question_prompt):
        """
        Sets the question prompt for the chatbot.
        """
        self.question_prompt = question_prompt

    def load_docs(self, docs_paths):
        """
        Loads the documents from the given paths.
        """
        texts = []
        for path in docs_paths:
            if path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.startswith('http'):
                loader = WebBaseLoader(path)
            else:
                loader = TextLoader(path)
            try:
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.splitter_chunk_size, chunk_overlap=self.splitter_chunk_overlap)
                texts += text_splitter.split_documents(documents)
            except Exception as e:
                print(f"Error loading documents from {path}: {e}")
        return texts

    def build_user_prompt(self, preprompt, question_prompt, question, relevant_docs=None, print_test=False):
        """
        Builds the user prompt.
        """
        prompt = f"{preprompt}\n"
        if relevant_docs is not None:
            for i, doc in enumerate(relevant_docs):

                if doc.metadata['source'].endswith('.pdf'):
                    prompt += f"""{doc.page_content}

source: {doc.metadata['source'].split("/")[-1].split(".")[0]}, page: {doc.metadata['page']}\n\n---
"""
                else:
                    prompt += f"""{doc.page_content}        

source: {doc.metadata['source'].split("/")[-1].split(".")[0]}\n\n---
"""

                #prompt += f"Document {i+1} (Source: {doc.metadata['source'].split("/")[-1].split(".")[0]}, page {doc.metadata['page']}):\n{doc.page_content}\n\n"
        prompt += f"\n{question_prompt}{question}"
        return prompt
    
    def ask(self, question):
        """
        Asks the chatbot a question.
        """
        if not self.retriever:
            return "Document retriever is not initialized."
        relevant_docs = self.retriever.get_relevant_documents(question)
        user_prompt = self.build_user_prompt(self.preprompt, 
                                             self.question_prompt, 
                                             question, 
                                             relevant_docs)
        self.add_message(user_prompt, role="user")
        try:
            ai_response = self.chat_openAI.invoke(self.chat_dict["messages"])
            self.add_message(ai_response.content, role="assistant")
        except Exception as e:
            print(f"Error invoking GPT chatbot: {e}")

        self.save_chat_to_json("Chats_Database/JSON")
        self.save_chat_to_md("Chats_Database/MD")

        return ai_response.content

# Implementation of the StandardRetrieverChatbot
class StandardRetrieverChatbot(RetrieverChatbot):
    def __init__(self, **kwargs):
        """
        Initializes the standard retriever chatbot.
        """
        super().__init__(**kwargs)
        if self.docs_paths and self.retriever is None:
            texts = self.load_docs(self.docs_paths)
            try:
                embeddings = OpenAIEmbeddings(model=self.embedding_model)
                db = FAISS.from_documents(texts, embeddings)
                self.retriever = db.as_retriever(search_kwargs={"k": self.k_docs})
            except Exception as e:
                print(f"Error during retriever initialization: {e}")
                self.retriever = None

# Implementation of the MultiQueryRetrieverChatbot
class MultiQueryRetrieverChatbot(RetrieverChatbot):
    """
    Chatbot that uses a MultiQueryRetriever with a LLM for retrieval.
    """
    def __init__(self, llm_temperature=0, **kwargs):
        super().__init__(**kwargs)
        if self.docs_paths and self.retriever is None:
            texts = self.load_docs(self.docs_paths)
            try:
                embeddings = OpenAIEmbeddings()
                vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)
                llm = ChatOpenAI(temperature=llm_temperature)
                self.retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(search_kwargs={"k": self.k_docs}), llm=llm)
                logging.basicConfig()
                logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
            except Exception as e:
                print(f"Error during retriever initialization: {e}")
                self.retriever = None

# Implementation of the MultiVectorRetrieverChatbot
class MultiVectorRetrieverChatbot(RetrieverChatbot):
    """
    Chatbot that uses a MultiVectorRetriever with a Chroma vectorstore for retrieval.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.docs_paths and self.retriever is None:
            texts = self.load_docs(self.docs_paths)
            try:
                vectorstore = Chroma(
                    collection_name="full_documents",
                    embedding_function=OpenAIEmbeddings(),
                )
                store = InMemoryByteStore()
                id_key = "doc_id"
                retriever = MultiVectorRetriever(
                    vectorstore=vectorstore,
                    byte_store=store,
                    id_key=id_key,
                    search_kwargs={"k": self.k_docs}

                )
                doc_ids = [str(uuid.uuid4()) for _ in texts]
                child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.splitter_chunk_size // 5, chunk_overlap=self.splitter_chunk_overlap // 5)
                sub_docs = []
                for i, doc in enumerate(texts):
                    _id = doc_ids[i]
                    _sub_docs = child_text_splitter.split_documents([doc])
                    for _doc in _sub_docs:
                        _doc.metadata[id_key] = _id
                    sub_docs.extend(_sub_docs)
                retriever.vectorstore.add_documents(sub_docs)
                retriever.docstore.mset(list(zip(doc_ids, texts)))
                self.retriever = retriever
            except Exception as e:
                print(f"Error during retriever initialization: {e}")
                self.retriever = None