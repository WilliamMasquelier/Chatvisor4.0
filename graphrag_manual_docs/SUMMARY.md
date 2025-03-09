# GraphRAG Documentation

## Home

* [Welcome](docs/index.md)
* [Getting Started](docs/get_started.md)
* [Development Guide](docs/developing.md)

## Indexing

* [Indexing Overview](docs/index/overview.md)
* [Architecture](docs/index/architecture.md)
* [Dataflow](docs/index/default_dataflow.md)
* [Outputs](docs/index/outputs.md)

## Prompt Tuning

* [Prompt Tuning Overview](docs/prompt_tuning/overview.md)
* [Auto Tuning](docs/prompt_tuning/auto_prompt_tuning.md)
* [Manual Tuning](docs/prompt_tuning/manual_prompt_tuning.md)

## Query

* [Query Overview](docs/query/overview.md)
* [Global Search](docs/query/global_search.md)
* [Local Search](docs/query/local_search.md)
* [DRIFT Search](docs/query/drift_search.md)
* [Question Generation](docs/query/question_generation.md)

## Query Notebooks

* [Notebooks Overview](docs/query/notebooks/overview.md)
* [Global Search Notebook](docs/examples_notebooks/global_search.md)
* [Local Search Notebook](docs/examples_notebooks/local_search.md)
* [DRIFT Search Notebook](docs/examples_notebooks/drift_search.md)

## Configuration

* [Configuration Overview](docs/config/overview.md)
* [Init Command](docs/config/init.md)
* [Using YAML](docs/config/yaml.md)
* [Using Env Vars](docs/config/env_vars.md)

## CLI

* [CLI](docs/cli.md)

## Extras

* [Microsoft Research Blog](docs/blog_posts.md)
* [Visualization Guide](docs/visualization_guide.md)
* [Operation Dulce - About](docs/data/operation_dulce/ABOUT.md)
* [Operation Dulce - Document](docs/data/operation_dulce/Operation%20Dulce%20v2%201%201.md)

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

Some unit and smoke tests use Azurite to emulate Azure resources. This can be started by running:

or by simply running azurite in the terminal if already installed globally. See the Azurite documentation for more information about how to install and use Azurite.

Our Python package utilizes Poetry to manage dependencies and poethepoet to manage build scripts.

Available scripts are:

Make sure llvm-9 and llvm-9-dev are installed:

sudo apt-get install llvm-9 llvm-9-dev

and then in your bashrc, add

export LLVM_CONFIG=/usr/bin/llvm-config-9

Make sure you have python3.10-dev installed or more generally python<version>-dev

sudo apt-get install python3.10-dev

GRAPHRAG_LLM_THREAD_COUNT and GRAPHRAG_EMBEDDING_THREAD_COUNT are both set to 50 by default. You can modify these values
to reduce concurrency. Please refer to the Configuration Documents

- poetry run poe index - Run the Indexing CLI
- poetry run poe query - Run the Query CLI
- poetry build - This invokes poetry build, which will build a wheel file and other distributable artifacts.
- poetry run poe test - This will execute all tests.
- poetry run poe test_unit - This will execute unit tests.
- poetry run poe test_integration - This will execute integration tests.
- poetry run poe test_smoke - This will execute smoke tests.
- poetry run poe test_verbs - This will execute tests of the basic workflows.
- poetry run poe check - This will perform a suite of static checks across the package, including:
- formatting
- documentation formatting
- linting
- security patterns
- type-checking
- poetry run poe fix - This will apply any available auto-fixes to the package. Usually this is just formatting fixes.
- poetry run poe fix_unsafe - This will apply any available auto-fixes to the package, including those that may be unsafe.
- poetry run poe format - Explicitly run the formatter across the package.

[Download](https://www.python.org/downloads/)

[Instructions](https://python-poetry.org/docs/#installation)

[](#__codelineno-0-1)

[](#__codelineno-0-2)

[](#__codelineno-1-1)

[](#__codelineno-2-1)

[](#__codelineno-3-1)

[Azurite documentation](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite)

[poethepoet](https://pypi.org/project/poethepoet/)

[Configuration Documents](https://microsoft.github.io/graphrag/../config/overview/)

# GraphRAG Indexing 🤖

## Getting Started

## Usage

## Further Reading

### Requirements

### CLI

### Python API

The GraphRAG indexing package is a data pipeline and transformation suite that is designed to extract meaningful, structured data from unstructured text using LLMs.

Indexing Pipelines are configurable. They are composed of workflows, standard and custom steps, prompt templates, and input/output adapters. Our standard pipeline is designed to:

The outputs of the pipeline are stored as Parquet tables by default, and embeddings are written to your configured vector store.

See the requirements section in Get Started for details on setting up a development environment.

To configure GraphRAG, see the configuration documentation.
After you have a config file you can run the pipeline using the CLI or the Python API.

Please see the indexing API python file for the recommended method to call directly from Python code.

- extract entities, relationships and claims from raw text
- perform community detection in entities
- generate community summaries and reports at multiple levels of granularity
- embed entities into a graph vector space
- embed text chunks into a textual vector space

- To start developing within the GraphRAG project, see getting started
- To understand the underlying concepts and execution model of the indexing library, see the architecture documentation
- To read more about configuring the indexing engine, see the configuration documentation

[requirements](https://microsoft.github.io/graphrag/../../developing/#requirements)

[Get Started](https://microsoft.github.io/graphrag/../../get_started/)

[configuration](https://microsoft.github.io/graphrag/../../config/overview/)

[](#__codelineno-0-1)

[](#__codelineno-0-2)

[python file](https://github.com/microsoft/graphrag/blob/main/graphrag/api/index.py)

[getting started](https://microsoft.github.io/graphrag/../../developing/)

[the architecture documentation](https://microsoft.github.io/graphrag/../architecture/)

[the configuration documentation](https://microsoft.github.io/graphrag/../../config/overview/)

In order to support the GraphRAG system, the outputs of the indexing engine (in the Default Configuration Mode) are aligned to a knowledge model we call the GraphRAG Knowledge Model.
This model is designed to be an abstraction over the underlying data storage technology, and to provide a common interface for the GraphRAG system to interact with.
In normal use-cases the outputs of the GraphRAG Indexer would be loaded into a database system, and the GraphRAG's Query Engine would interact with the database using the knowledge model data-store types.

Because of the complexity of our data indexing tasks, we needed to be able to express our data pipeline as series of multiple, interdependent workflows.

The GraphRAG library was designed with LLM interactions in mind, and a common setback when working with LLM APIs is various errors due to network latency, throttling, etc..
Because of these potential error cases, we've added a cache layer around LLM interactions.
When completion requests are made using the same input set (prompt and tuning parameters), we return a cached result if one exists.
This allows our indexer to be more resilient to network issues, to act idempotently, and to provide a more efficient end-user experience.

The knowledge model is a specification for data outputs that conform to our data-model definition. You can find these definitions in the python/graphrag/graphrag/model folder within the GraphRAG repository. The following entity types are provided. The fields here represent the fields that are text-embedded by default.

Let's take a look at how the default-configuration workflow transforms text documents into the GraphRAG Knowledge Model. This page gives a general overview of the major steps in this process. To fully configure this workflow, check out the configuration documentation.

The first phase of the default-configuration workflow is to transform input documents into TextUnits. A TextUnit is a chunk of text that is used for our graph extraction techniques. They are also used as source-references by extracted knowledge items in order to empower breadcrumbs and provenance by concepts back to their original source text.

The chunk size (counted in tokens), is user-configurable. By default this is set to 300 tokens, although we've had positive experience with 1200-token chunks using a single "glean" step. (A "glean" step is a follow-on extraction). Larger chunks result in lower-fidelity output and less meaningful reference texts; however, using larger chunks can result in much faster processing time.

The group-by configuration is also user-configurable. By default, we align our chunks to document boundaries, meaning that there is a strict 1-to-many relationship between Documents and TextUnits. In rare cases, this can be turned into a many-to-many relationship. This is useful when the documents are very short and we need several of them to compose a meaningful analysis unit (e.g. Tweets or a chat log)

In this phase, we analyze each text unit and extract our graph primitives: Entities, Relationships, and Claims.
Entities and Relationships are extracted at once in our entity_extract verb, and claims are extracted in our claim_extract verb. Results are then combined and passed into following phases of the pipeline.

In this first step of graph extraction, we process each text-unit in order to extract entities and relationships out of the raw text using the LLM. The output of this step is a subgraph-per-TextUnit containing a list of entities with a title, type, and description, and a list of relationships with a source, target, and description.

These subgraphs are merged together - any entities with the same title and type are merged by creating an array of their descriptions. Similarly, any relationships with the same source and target are merged by creating an array of their descriptions.

Now that we have a graph of entities and relationships, each with a list of descriptions, we can summarize these lists into a single description per entity and relationship. This is done by asking the LLM for a short summary that captures all of the distinct information from each description. This allows all of our entities and relationships to have a single concise description.

Finally, as an independent workflow, we extract claims from the source TextUnits. These claims represent positive factual statements with an evaluated status and time-bounds. These get exported as a primary artifact called Covariates.

Note: claim extraction is optional and turned off by default. This is because claim extraction generally requires prompt tuning to be useful.

Now that we have a usable graph of entities and relationships, we want to understand their community structure. These give us explicit ways of understanding the topological structure of our graph.

In this step, we generate a hierarchy of entity communities using the Hierarchical Leiden Algorithm. This method will apply a recursive community-clustering to our graph until we reach a community-size threshold. This will allow us to understand the community structure of our graph and provide a way to navigate and summarize the graph at different levels of granularity.

Once our graph augmentation steps are complete, the final Entities, Relationships, and Communities tables are exported.

At this point, we have a functional graph of entities and relationships and a hierarchy of communities for the entities.

Now we want to build on the communities data and generate reports for each community. This gives us a high-level understanding of the graph at several points of graph granularity. For example, if community A is the top-level community, we'll get a report about the entire graph. If the community is lower-level, we'll get a report about a local cluster.

In this step, we generate a summary of each community using the LLM. This will allow us to understand the distinct information contained within each community and provide a scoped understanding of the graph, from either a high-level or a low-level perspective. These reports contain an executive overview and reference the key entities, relationships, and claims within the community sub-structure.

In this step, each community report is then summarized via the LLM for shorthand use.

At this point, some bookkeeping work is performed and we export the Community Reports tables.

In this phase of the workflow, we create the Documents table for the knowledge model.

If the workflow is operating on CSV data, you may configure your workflow to add additional fields to Documents output. These fields should exist on the incoming CSV tables. Details about configuring this can be found in the configuration documentation.

In this step, we link each document to the text-units that were created in the first phase. This allows us to understand which documents are related to which text-units and vice-versa.

At this point, we can export the Documents table into the knowledge Model.

In this phase of the workflow, we perform some steps to support network visualization of our high-dimensional vector spaces within our existing graphs. At this point there are two logical graphs at play: the Entity-Relationship graph and the Document graph.

In this step, we generate a vector representation of our graph using the Node2Vec algorithm. This will allow us to understand the implicit structure of our graph and provide an additional vector-space in which to search for related concepts during our query phase.

For each of the logical graphs, we perform a UMAP dimensionality reduction to generate a 2D representation of the graph. This will allow us to visualize the graph in a 2D space and understand the relationships between the nodes in the graph. The UMAP embeddings are reduced to two dimensions as x/y coordinates.

For all artifacts that require downstream vector search, we generate text embeddings as a final step. These embeddings are written directly to a configured vector store. By default we embed entity descriptions, text unit text, and community report text.

- Document - An input document into the system. These either represent individual rows in a CSV or individual .txt file.
- TextUnit - A chunk of text to analyze. The size of these chunks, their overlap, and whether they adhere to any data boundaries may be configured below. A common use case is to set CHUNK_BY_COLUMNS to id so that there is a 1-to-many relationship between documents and TextUnits instead of a many-to-many.
- Entity - An entity extracted from a TextUnit. These represent people, places, events, or some other entity-model that you provide.
- Relationship - A relationship between two entities.
- Covariate - Extracted claim information, which contains statements about entities which may be time-bound.
- Community - Once the graph of entities and relationships is built, we perform hierarchical community detection on them to create a clustering structure.
- Community Report - The contents of each community are summarized into a generated report, useful for human reading and downstream search.

[configuration](https://microsoft.github.io/graphrag/../../config/overview/)

[configuration documentation](https://microsoft.github.io/graphrag/../../config/overview/)

