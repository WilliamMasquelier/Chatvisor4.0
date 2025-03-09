# Manual Prompt Tuning ⚙️

## Indexing Prompts

## Query Prompts

### Entity/Relationship Extraction

### Summarize Entity/Relationship Descriptions

### Claim Extraction

### Generate Community Reports

### Local Search

### Global Search

### Drift Search

#### Tokens

#### Tokens

#### Tokens

#### Tokens

#### Tokens

#### Tokens

#### Tokens

The GraphRAG indexer, by default, will run with a handful of prompts that are designed to work well in the broad context of knowledge discovery.
However, it is quite common to want to tune the prompts to better suit your specific use case.
We provide a means for you to do this by allowing you to specify a custom prompt file, which will each use a series of token-replacements internally.

Each of these prompts may be overridden by writing a custom prompt file in plaintext. We use token-replacements in the form of {token_name}, and the descriptions for the available tokens can be found below.

Prompt Source

Prompt Source

Prompt Source

See the configuration documentation for details on how to change this.

Prompt Source

Prompt Source

Mapper Prompt Source

Reducer Prompt Source

Knowledge Prompt Source

Global search uses a map/reduce approach to summarization. You can tune these prompts independently. This search also includes the ability to adjust the use of general knowledge from the model's training.

Prompt Source

- {input_text} - The input text to be processed.
- {entity_types} - A list of entity types
- {tuple_delimiter} - A delimiter for separating values within a tuple. A single tuple is used to represent an individual entity or relationship.
- {record_delimiter} - A delimiter for separating tuple instances.
- {completion_delimiter} - An indicator for when generation is complete.

- {entity_name} - The name of the entity or the source/target pair of the relationship.
- {description_list} - A list of descriptions for the entity or relationship.

- {input_text} - The input text to be processed.
- {tuple_delimiter} - A delimiter for separating values within a tuple. A single tuple is used to represent an individual entity or relationship.
- {record_delimiter} - A delimiter for separating tuple instances.
- {completion_delimiter} - An indicator for when generation is complete.
- {entity_specs} - A list of entity types.
- {claim_description} - Description of what claims should look like. Default is: "Any claims or facts that could be relevant to information discovery."

- {input_text} - The input text to generate the report with. This will contain tables of entities and relationships.

- {response_type} - Describe how the response should look. We default to "multiple paragraphs".
- {context_data} - The data tables from GraphRAG's index.

- {response_type} - Describe how the response should look (reducer only). We default to "multiple paragraphs".
- {context_data} - The data tables from GraphRAG's index.

- {response_type} - Describe how the response should look. We default to "multiple paragraphs".
- {context_data} - The data tables from GraphRAG's index.
- {community_reports} - The most relevant community reports to include in the summarization.
- {query} - The query text as injected into the context.

[Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/index/entity_extraction.py)

[Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/index/summarize_descriptions.py)

[Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/index/claim_extraction.py)

[configuration documentation](https://microsoft.github.io/graphrag/../../config/overview/)

[Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/index/community_report.py)

[Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/query/local_search_system_prompt.py)

[Mapper Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/query/global_search_map_system_prompt.py)

[Reducer Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/query/global_search_reduce_system_prompt.py)

[Knowledge Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/query/global_search_knowledge_system_prompt.py)

[Prompt Source](http://github.com/microsoft/graphrag/blob/main/graphrag/prompts/query/drift_search_system_prompt.py)

