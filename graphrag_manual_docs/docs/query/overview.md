# Query Engine 🔎

## Local Search

## Global Search

## DRIFT Search

## Question Generation

The Query Engine is the retrieval module of the Graph RAG Library. It is one of the two main components of the Graph RAG library, the other being the Indexing Pipeline (see Indexing Pipeline).
It is responsible for the following tasks:

Local search method generates answers by combining relevant data from the AI-extracted knowledge-graph with text chunks of the raw documents. This method is suitable for questions that require an understanding of specific entities mentioned in the documents (e.g. What are the healing properties of chamomile?).

For more details about how Local Search works please refer to the Local Search documentation.

Global search method generates answers by searching over all AI-generated community reports in a map-reduce fashion. This is a resource-intensive method, but often gives good responses for questions that require an understanding of the dataset as a whole (e.g. What are the most significant values of the herbs mentioned in this notebook?).

More about this can be checked at the Global Search documentation.

DRIFT Search introduces a new approach to local search queries by including community information in the search process. This greatly expands the breadth of the query’s starting point and leads to retrieval and usage of a far higher variety of facts in the final answer. This addition expands the GraphRAG query engine by providing a more comprehensive option for local search, which uses community insights to refine a query into detailed follow-up questions.

To learn more about DRIFT Search, please refer to the DRIFT Search documentation.

This functionality takes a list of user queries and generates the next candidate questions. This is useful for generating follow-up questions in a conversation or for generating a list of questions for the investigator to dive deeper into the dataset.

Information about how question generation works can be found at the Question Generation documentation page.

- Local Search
- Global Search
- DRIFT Search
- Question Generation

[Indexing Pipeline](https://microsoft.github.io/graphrag/../../index/overview/)

[Local Search](#local-search)

[Global Search](#global-search)

[DRIFT Search](#drift-search)

[Question Generation](#question-generation)

[Local Search](https://microsoft.github.io/graphrag/../local_search/)

[Global Search](https://microsoft.github.io/graphrag/../global_search/)

[DRIFT Search](https://microsoft.github.io/graphrag/../drift_search/)

[Question Generation](https://microsoft.github.io/graphrag/../question_generation/)

