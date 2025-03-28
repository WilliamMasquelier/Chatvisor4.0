# Default Configuration Mode (using YAML/JSON)

# Config Sections

## Indexing

## Query

### models

### embed_text

### vector_store

### input

### chunks

### cache

### output

### update_index_storage

### reporting

### extract_graph

### summarize_descriptions

### extract_graph_nlp

### extract_claims

### community_reports

### prune_graph

### cluster_graph

### embed_graph

### umap

### snapshots

### local_search

### global_search

### drift_search

### basic_search

### workflows

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

#### Fields

The default configuration mode may be configured by using a settings.yml or settings.json file in the data project root. If a .env file is present along with this config file, then it will be loaded, and the environment variables defined therein will be available for token replacements in your configuration document using ${ENV_VAR} syntax. We initialize with YML by default in graphrag init but you may use the equivalent JSON form if preferred.

Many of these config values have defaults. Rather than replicate them here, please refer to the constants in the code directly.

For example:

This is a dict of model configurations. The dict key is used to reference this configuration elsewhere when a model instance is desired. In this way, you can specify as many different models as you need, and reference them differentially in the workflow steps.

For example:
models:
  default_chat_model:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_chat
    model: gpt-4o
    model_supports_json: true
  default_embedding_model:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-ada-002

By default, the GraphRAG indexer will only export embeddings required for our query methods. However, the model has embeddings defined for all plaintext fields, and these can be customized by setting the target and names fields.

Supported embeddings names are:
- text_unit.text
- document.text
- entity.title
- entity.description
- relationship.description
- community.title
- community.summary
- community.full_content

Where to put all vectors for the system. Configured for lancedb by default.

Our pipeline can ingest .csv or .txt data from an input folder. These files can be nested within subfolders. In general, CSV-based data provides the most customizability. Each CSV should at least contain a text field. You can use the metadata list to specify additional columns from the CSV to include as headers in each text chunk, allowing you to repeat document content within each chunk for better LLM inclusion.

These settings configure how we parse documents into text chunks. This is necessary because very large documents may not fit into a single context window, and graph extraction accuracy can be modulated. Also note the metadata setting in the input document config, which will replicate document metadata into each chunk.

This section controls the cache mechanism used by the pipeline. This is used to cache LLM invocation results.

This section controls the storage mechanism used by the pipeline used for exporting output tables.

The section defines a secondary storage location for running incremental indexing, to preserve your original outputs.

This section controls the reporting mechanism used by the pipeline, for common events and error messages. The default is to write reports to a file in the output directory. However, you can also choose to write reports to the console or to an Azure Blob Storage container.

Defines settings for NLP-based graph extraction methods.

Parameters for manual graph pruning. This can be used to optimize the modularity of your graph clusters, by removing overly-connected or rare nodes.

These are the settings used for Leiden hierarchical clustering of the graph to create communities.

We use node2vec to embed the graph. This is primarily used for visualization, so it is not turned on by default. However, if you do prefer to embed the graph for secondary analysis, you can turn this on and we will persist the embeddings to your configured vector store.

Indicates whether we should run UMAP dimensionality reduction. This is used to provide an x/y coordinate to each graph node, suitable for visualization. If this is not enabled, nodes will receive a 0/0 x/y coordinate. If this is enabled, you must enable graph embedding as well.

list[str] - This is a list of workflow names to run, in order. GraphRAG has built-in pipelines to configure this, but you can run exactly and only what you want by specifying the list here. Useful if you have done part of the processing yourself.

- api_key str - The OpenAI API key to use.
- type openai_chat|azure_openai_chat|openai_embedding|azure_openai_embedding - The type of LLM to use.
- model str - The model name.
- encoding_model str - The text encoding model to use. Default is to use the encoding model aligned with the language model (i.e., it is retrieved from tiktoken if unset).
- max_tokens int - The maximum number of output tokens.
- request_timeout float - The per-request timeout.
- api_base str - The API base url to use.
- api_version str - The API version.
- organization str - The client organization.
- proxy str - The proxy URL to use.
- azure_auth_type api_key|managed_identity - if using Azure, please indicate how you want to authenticate requests.
- audience str - (Azure OpenAI only) The URI of the target Azure resource/service for which a managed identity token is requested. Used if api_key is not defined. Default=https://cognitiveservices.azure.com/.default
- deployment_name str - The deployment name to use (Azure).
- model_supports_json bool - Whether the model supports JSON-mode output.
- tokens_per_minute int - Set a leaky-bucket throttle on tokens-per-minute.
- requests_per_minute int - Set a leaky-bucket throttle on requests-per-minute.
- max_retries int - The maximum number of retries to use.
- max_retry_wait float - The maximum backoff time.
- sleep_on_rate_limit_recommendation bool - Whether to adhere to sleep recommendations (Azure).
- concurrent_requests int The number of open requests to allow at once.
- temperature float - The temperature to use.
- top_p float - The top-p value to use.
- n int - The number of completions to generate.
- parallelization_stagger float - The threading stagger value.
- parallelization_num_threads int - The maximum number of work threads.
- async_mode asyncio|threaded The async mode to use. Either asyncio or `threaded.

- model_id str - Name of the model definition to use for text embedding.
- batch_size int - The maximum batch size to use.
- batch_max_tokens int - The maximum batch # of tokens.
- target required|all|selected|none - Determines which set of embeddings to export.
- names list[str] - If target=selected, this should be an explicit list of the embeddings names we support.

- type str - lancedb or azure_ai_search. Default=lancedb
- db_uri str (only for lancedb) - The database uri. Default=storage.base_dir/lancedb
- url str (only for AI Search) - AI Search endpoint
- api_key str (optional - only for AI Search) - The AI Search api key to use.
- audience str (only for AI Search) - Audience for managed identity token if managed identity authentication is used.
- overwrite bool (only used at index creation time) - Overwrite collection if it exist. Default=True
- container_name str - The name of a vector container. This stores all indexes (tables) for a given dataset ingest. Default=default

- type file|blob - The input type to use. Default=file
- file_type text|csv - The type of input data to load. Either text or csv. Default is text
- base_dir str - The base directory to read input from, relative to the root.
- connection_string str - (blob only) The Azure Storage connection string.
- storage_account_blob_url str - The storage account blob URL to use.
- container_name str - (blob only) The Azure Storage container name.
- file_encoding str - The encoding of the input file. Default is utf-8
- file_pattern str - A regex to match input files. Default is .*\.csv$ if in csv mode and .*\.txt$ if in text mode.
- file_filter dict - Key/value pairs to filter. Default is None.
- text_column str - (CSV Mode Only) The text column name.
- metadata list[str] - (CSV Mode Only) The additional document attributes to include.

- size int - The max chunk size in tokens.
- overlap int - The chunk overlap in tokens.
- group_by_columns list[str] - group documents by fields before chunking.
- encoding_model str - The text encoding model to use for splitting on token boundaries.
- prepend_metadata bool - Determines if metadata values should be added at the beginning of each chunk. Default=False.
- chunk_size_includes_metadata bool - Specifies whether the chunk size calculation should include metadata tokens. Default=False.

- type file|memory|none|blob - The cache type to use. Default=file
- connection_string str - (blob only) The Azure Storage connection string.
- container_name str - (blob only) The Azure Storage container name.
- base_dir str - The base directory to write cache to, relative to the root.
- storage_account_blob_url str - The storage account blob URL to use.

- type file|memory|blob - The storage type to use. Default=file
- connection_string str - (blob only) The Azure Storage connection string.
- container_name str - (blob only) The Azure Storage container name.
- base_dir str - The base directory to write output artifacts to, relative to the root.
- storage_account_blob_url str - The storage account blob URL to use.

- type file|memory|blob - The storage type to use. Default=file
- connection_string str - (blob only) The Azure Storage connection string.
- container_name str - (blob only) The Azure Storage container name.
- base_dir str - The base directory to write output artifacts to, relative to the root.
- storage_account_blob_url str - The storage account blob URL to use.

- type file|console|blob - The reporting type to use. Default=file
- connection_string str - (blob only) The Azure Storage connection string.
- container_name str - (blob only) The Azure Storage container name.
- base_dir str - The base directory to write reports to, relative to the root.
- storage_account_blob_url str - The storage account blob URL to use.

- model_id str - Name of the model definition to use for API calls.
- prompt str - The prompt file to use.
- entity_types list[str] - The entity types to identify.
- max_gleanings int - The maximum number of gleaning cycles to use.

- model_id str - Name of the model definition to use for API calls.
- prompt str - The prompt file to use.
- max_length int - The maximum number of output tokens per summarization.

- normalize_edge_weights bool - Whether to normalize the edge weights during graph construction. Default=True.
- text_analyzer dict - Parameters for the NLP model.
- extractor_type regex_english|syntactic_parser|cfg - Default=regex_english.
- model_name str - Name of NLP model (for SpaCy-based models)
- max_word_length int - Longest word to allow. Default=15.
- word_delimiter str - Delimiter to split words. Default ' '.
- include_named_entities bool - Whether to include named entities in noun phrases. Default=True.
- exclude_nouns list[str] | None - List of nouns to exclude. If None, we use an internal stopword list.
- exclude_entity_tags list[str] - List of entity tags to ignore.
- exclude_pos_tags list[str] - List of part-of-speech tags to ignore.
- noun_phrase_tags list[str] - List of noun phrase tags to ignore.
- noun_phrase_grammars dict[str, str] - Noun phrase grammars for the model (cfg-only).

- enabled bool - Whether to enable claim extraction. Off by default, because claim prompts really need user tuning.
- model_id str - Name of the model definition to use for API calls.
- prompt str - The prompt file to use.
- description str - Describes the types of claims we want to extract.
- max_gleanings int - The maximum number of gleaning cycles to use.

- model_id str - Name of the model definition to use for API calls.
- prompt str - The prompt file to use.
- max_length int - The maximum number of output tokens per report.
- max_input_length int - The maximum number of input tokens to use when generating reports.

- min_node_freq int - The minimum node frequency to allow.
- max_node_freq_std float | None - The maximum standard deviation of node frequency to allow.
- min_node_degree int - The minimum node degree to allow.
- max_node_degree_std float | None - The maximum standard deviation of node degree to allow.
- min_edge_weight_pct int - The minimum edge weight percentile to allow.
- remove_ego_nodes bool - Remove ego nodes.
- lcc_only bool - Only use largest connected component.

- max_cluster_size int - The maximum cluster size to export.
- use_lcc bool - Whether to only use the largest connected component.
- seed int - A randomization seed to provide if consistent run-to-run results are desired. We do provide a default in order to guarantee clustering stability.

- enabled bool - Whether to enable graph embeddings.
- num_walks int - The node2vec number of walks.
- walk_length int - The node2vec walk length.
- window_size int - The node2vec window size.
- iterations int - The node2vec number of iterations.
- random_seed int - The node2vec random seed.
- strategy dict - Fully override the embed graph strategy.

- enabled bool - Whether to enable UMAP layouts.

- embeddings bool - Export embeddings snapshots to parquet.
- graphml bool - Export graph snapshots to GraphML.

- chat_model_id str - Name of the model definition to use for Chat Completion calls.
- embedding_model_id str - Name of the model definition to use for Embedding calls.
- prompt str - The prompt file to use.
- text_unit_prop float - The text unit proportion.
- community_prop float - The community proportion.
- conversation_history_max_turns int - The conversation history maximum turns.
- top_k_entities int - The top k mapped entities.
- top_k_relationships int - The top k mapped relations.
- temperature float | None - The temperature to use for token generation.
- top_p float | None - The top-p value to use for token generation.
- n int | None - The number of completions to generate.
- max_tokens int - The maximum tokens.
- llm_max_tokens int - The LLM maximum tokens.

- chat_model_id str - Name of the model definition to use for Chat Completion calls.
- map_prompt str - The mapper prompt file to use.
- reduce_prompt str - The reducer prompt file to use.
- knowledge_prompt str - The knowledge prompt file to use.
- map_prompt str | None - The global search mapper prompt to use.
- reduce_prompt str | None - The global search reducer to use.
- knowledge_prompt str | None - The global search general prompt to use.
- temperature float | None - The temperature to use for token generation.
- top_p float | None - The top-p value to use for token generation.
- n int | None - The number of completions to generate.
- max_tokens int - The maximum context size in tokens.
- data_max_tokens int - The data llm maximum tokens.
- map_max_tokens int - The map llm maximum tokens.
- reduce_max_tokens int - The reduce llm maximum tokens.
- concurrency int - The number of concurrent requests.
- dynamic_search_llm str - LLM model to use for dynamic community selection.
- dynamic_search_threshold int - Rating threshold in include a community report.
- dynamic_search_keep_parent bool - Keep parent community if any of the child communities are relevant.
- dynamic_search_num_repeats int - Number of times to rate the same community report.
- dynamic_search_use_summary bool - Use community summary instead of full_context.
- dynamic_search_concurrent_coroutines int - Number of concurrent coroutines to rate community reports.
- dynamic_search_max_level int - The maximum level of community hierarchy to consider if none of the processed communities are relevant.

- chat_model_id str - Name of the model definition to use for Chat Completion calls.
- embedding_model_id str - Name of the model definition to use for Embedding calls.
- prompt str - The prompt file to use.
- reduce_prompt str - The reducer prompt file to use.
- temperature float - The temperature to use for token generation.",
- top_p float - The top-p value to use for token generation.
- n int - The number of completions to generate.
- max_tokens int - The maximum context size in tokens.
- data_max_tokens int - The data llm maximum tokens.
- concurrency int - The number of concurrent requests.
- drift_k_followups int - The number of top global results to retrieve.
- primer_folds int - The number of folds for search priming.
- primer_llm_max_tokens int - The maximum number of tokens for the LLM in primer.
- n_depth int - The number of drift search steps to take.
- local_search_text_unit_prop float - The proportion of search dedicated to text units.
- local_search_community_prop float - The proportion of search dedicated to community properties.
- local_search_top_k_mapped_entities int - The number of top K entities to map during local search.
- local_search_top_k_relationships int - The number of top K relationships to map during local search.
- local_search_max_data_tokens int - The maximum context size in tokens for local search.
- local_search_temperature float - The temperature to use for token generation in local search.
- local_search_top_p float - The top-p value to use for token generation in local search.
- local_search_n int - The number of completions to generate in local search.
- local_search_llm_max_gen_tokens int - The maximum number of generated tokens for the LLM in local search.

- chat_model_id str - Name of the model definition to use for Chat Completion calls.
- embedding_model_id str - Name of the model definition to use for Embedding calls.
- prompt str - The prompt file to use.
- text_unit_prop float - The text unit proportion.
- community_prop float - The community proportion.
- conversation_history_max_turns int - The conversation history maximum turns.
- top_k_entities int - The top k mapped entities.
- top_k_relationships int - The top k mapped relations.
- temperature float | None - The temperature to use for token generation.
- top_p float | None - The top-p value to use for token generation.
- n int | None - The number of completions to generate.
- max_tokens int - The maximum tokens.
- llm_max_tokens int - The LLM maximum tokens.

[constants in the code](https://github.com/microsoft/graphrag/blob/main/graphrag/config/defaults.py)

[](#__codelineno-0-1)

[](#__codelineno-0-2)

[](#__codelineno-0-3)

[](#__codelineno-0-4)

[](#__codelineno-0-5)

[](#__codelineno-0-6)

[](#__codelineno-1-1)

[](#__codelineno-1-2)

[](#__codelineno-1-3)

[](#__codelineno-1-4)

[](#__codelineno-1-5)

[](#__codelineno-1-6)

[](#__codelineno-1-7)

[](#__codelineno-1-8)

[](#__codelineno-1-9)

[](#__codelineno-1-10)

