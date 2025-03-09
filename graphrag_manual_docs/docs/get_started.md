# Getting Started

# Overview

# Install GraphRAG

# Running the Indexer

# Using the Query Engine

# Visualizing the Graph

## Requirements

## Quickstart

## Set Up Your Workspace Variables

## Running the Indexing pipeline

## Running the Query Engine

#### OpenAI and Azure OpenAI

#### Azure OpenAI

Python 3.10-3.12

To get started with the GraphRAG system, you have a few options:

👉 Use the GraphRAG Accelerator solution 
👉 Install from pypi. 
👉 Use it from source

To get started with the GraphRAG system we recommend trying the Solution Accelerator package. This provides a user-friendly end-to-end experience with Azure resources.

The following is a simple end-to-end example for using the GraphRAG system.
It shows how to use the system to index some text, and then use the indexed data to answer questions about the documents.

The graphrag library includes a CLI for a no-code approach to getting started. Please review the full CLI documentation for further detail.

We need to set up a data project and some initial configuration. First let's get a sample dataset ready:

Get a copy of A Christmas Carol by Charles Dickens from a trusted source:

To initialize your workspace, first run the graphrag init command.
Since we have already configured a directory named ./ragtest in the previous step, run the following command:

This will create two files: .env and settings.yaml in the ./ragtest directory.

If running in OpenAI mode, update the value of GRAPHRAG_API_KEY in the .env file with your OpenAI API key.

In addition, Azure OpenAI users should set the following variables in the settings.yaml file. To find the appropriate sections, just search for the llm: configuration, you should see two sections, one for the chat endpoint and one for the embeddings endpoint. Here is an example of how to configure the chat endpoint:

Finally we'll run the pipeline!



This process will take some time to run. This depends on the size of your input data, what model you're using, and the text chunk size being used (these can be configured in your settings.yml file).
Once the pipeline is complete, you should see a new folder called ./ragtest/output with a series of parquet files.

Now let's ask some questions using this dataset.

Here is an example using Global search to ask a high-level question:

Here is an example using Local search to ask a more specific question about a particular character:

Please refer to Query Engine docs for detailed information about how to leverage our Local and Global search mechanisms for extracting meaningful insights from data after the Indexer has wrapped up execution.

Check out our visualization guide for a more interactive experience in debugging and exploring the knowledge graph.

- .env contains the environment variables required to run the GraphRAG pipeline. If you inspect the file, you'll see a single environment variable defined,
  GRAPHRAG_API_KEY=<API_KEY>. This is the API key for the OpenAI API or Azure OpenAI endpoint. You can replace this with your own API key. If you are using another form of authentication (i.e. managed identity), please delete this file.
- settings.yaml contains the settings for the pipeline. You can modify this file to change the settings for the pipeline.

- For more details about configuring GraphRAG, see the configuration documentation.
- To learn more about Initialization, refer to the Initialization documentation.
- For more details about using the CLI, refer to the CLI documentation.

[Python 3.10-3.12](https://www.python.org/downloads/)

[Use the GraphRAG Accelerator solution](https://github.com/Azure-Samples/graphrag-accelerator)

[Install from pypi](https://pypi.org/project/graphrag/)

[Use it from source](https://microsoft.github.io/graphrag/../developing/)

[Solution Accelerator](https://github.com/Azure-Samples/graphrag-accelerator)

[](#__codelineno-0-1)

[CLI documentation](https://microsoft.github.io/graphrag/../cli/)

[](#__codelineno-1-1)

[](#__codelineno-2-1)

[](#__codelineno-3-1)

[](#__codelineno-4-1)

[](#__codelineno-4-2)

[](#__codelineno-4-3)

[](#__codelineno-4-4)

[configuration documentation](https://microsoft.github.io/graphrag/../config/overview/)

[Initialization documentation](https://microsoft.github.io/graphrag/../config/init/)

[CLI documentation](https://microsoft.github.io/graphrag/../cli/)

[](#__codelineno-5-1)

[](#__codelineno-6-1)

[](#__codelineno-6-2)

[](#__codelineno-6-3)

[](#__codelineno-6-4)

[](#__codelineno-7-1)

[](#__codelineno-7-2)

[](#__codelineno-7-3)

[](#__codelineno-7-4)

[Query Engine](https://microsoft.github.io/graphrag/../query/overview/)

[visualization guide](https://microsoft.github.io/graphrag/../visualization_guide/)

![pipeline executing from the CLI](https://microsoft.github.io/graphrag/../img/pipeline-running.png)

