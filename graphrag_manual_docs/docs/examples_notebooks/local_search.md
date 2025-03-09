# Local Search

## Local Search Example¶

### Load text units and graph data tables as context for local search¶

### Load tables to dataframes¶

### Create local search context builder¶

### Create local search engine¶

### Run local search on sample queries¶

### Question Generation¶

#### Read entities¶

#### Read relationships¶

#### Read community reports¶

#### Read text units¶

#### Inspecting the context data used to generate the response¶

Local search method generates answers by combining relevant data from the AI-extracted knowledge-graph with text chunks of the raw documents. This method is suitable for questions that require an understanding of specific entities mentioned in the documents (e.g. What are the healing properties of chamomile?).

This function takes a list of user queries and generates the next candidate questions.

- In this test we first load indexing outputs from parquet files to dataframes, then convert these dataframes into collections of data objects aligning with the knowledge model.

[¶](#local-search-example)

[¶](#load-text-units-and-graph-data-tables-as-context-for-local-search)

[¶](#load-tables-to-dataframes)

[¶](#read-entities)

[¶](#read-relationships)

[¶](#read-community-reports)

[¶](#read-text-units)

[¶](#create-local-search-context-builder)

[¶](#create-local-search-engine)

[¶](#run-local-search-on-sample-queries)

[¶](#inspecting-the-context-data-used-to-generate-the-response)

[¶](#question-generation)

