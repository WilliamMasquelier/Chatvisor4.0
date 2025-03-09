# Question Generation ❔

## Entity-based Question Generation

## Methodology

## Configuration

## How to Use

The question generation method combines structured data from the knowledge graph with unstructured data from the input documents to generate candidate questions related to specific entities.

Given a list of prior user questions, the question generation method uses the same context-building approach employed in local search to extract and prioritize relevant structured and unstructured data, including entities, relationships, covariates, community reports and raw text chunks. These data records are then fitted into a single LLM prompt to generate candidate follow-up questions that represent the most important or urgent information content or themes in the data.

Below are the key parameters of the Question Generation class:

An example of the question generation function can be found in the following notebook.

- llm: OpenAI model object to be used for response generation
- context_builder: context builder object to be used for preparing context data from collections of knowledge model objects, using the same context builder class as in local search
- system_prompt: prompt template used to generate candidate questions. Default template can be found at system_prompt
- llm_params: a dictionary of additional parameters (e.g., temperature, max_tokens) to be passed to the LLM call
- context_builder_params: a dictionary of additional parameters to be passed to the context_builder object when building context for the question generation prompt
- callbacks: optional callback functions, can be used to provide custom event handlers for LLM's completion streaming events

[question generation](https://github.com/microsoft/graphrag/blob/main//graphrag/query/question_gen/)

[local search](https://microsoft.github.io/graphrag/../local_search/)

[Question Generation class](https://github.com/microsoft/graphrag/blob/main//graphrag/query/question_gen/local_gen.py)

[context builder](https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/local_search/mixed_context.py)

[system_prompt](https://github.com/microsoft/graphrag/blob/main//graphrag/prompts/query/question_gen_system_prompt.py)

[context_builder](https://github.com/microsoft/graphrag/blob/main//graphrag/query/structured_search/local_search/mixed_context.py)

[notebook](https://microsoft.github.io/graphrag/../../examples_notebooks/local_search/)

