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

