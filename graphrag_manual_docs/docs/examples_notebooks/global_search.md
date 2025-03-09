# Global Search

## Global Search example¶

### LLM setup¶

### Load community reports as context for global search¶

#### Build global context based on community reports¶

#### Perform global search¶

Global search method generates answers by searching over all AI-generated community reports in a map-reduce fashion. This is a resource-intensive method, but often gives good responses for questions that require an understanding of the dataset as a whole (e.g. What are the most significant values of the herbs mentioned in this notebook?).

- Load all community reports in the community_reports table from GraphRAG, to be used as context data for global search.
- Load entities from the entities tables from GraphRAG, to be used for calculating community weights for context ranking. Note that this is optional (if no entities are provided, we will not calculate community weights and only use the rank attribute in the community reports table for context ranking)
- Load all communities in the communities table from the GraphRAG, to be used to reconstruct the community graph hierarchy for dynamic community selection.

[¶](#global-search-example)

[¶](#llm-setup)

[¶](#load-community-reports-as-context-for-global-search)

[¶](#build-global-context-based-on-community-reports)

[¶](#perform-global-search)

