# Configuring GraphRAG Indexing

## Usage

## Options

## Example

## Output

## Next Steps

To start using GraphRAG, you must generate a configuration file. The init command is the easiest way to get started. It will create a .env and settings.yaml files in the specified directory with the necessary configuration settings. It will also output the default LLM prompts used by GraphRAG.

The init command will create the following files in the specified directory:

After initializing your workspace, you can either run the Prompt Tuning command to adapt the prompts to your data or even start running the Indexing Pipeline to index your data. For more information on configuring GraphRAG, see the Configuration documentation.

- --root PATH - The project root directory to initialize graphrag at. Default is the current directory.
- --force, --no-force - Optional, default is --no-force. Overwrite existing configuration and prompt files if they exist.

- settings.yaml - The configuration settings file. This file contains the configuration settings for GraphRAG.
- .env - The environment variables file. These are referenced in the settings.yaml file.
- prompts/ - The LLM prompts folder. This contains the default prompts used by GraphRAG, you can modify them or run the Auto Prompt Tuning command to generate new prompts adapted to your data.

[](#__codelineno-0-1)

[](#__codelineno-1-1)

[Auto Prompt Tuning](https://microsoft.github.io/graphrag/../../prompt_tuning/auto_prompt_tuning/)

[Prompt Tuning](https://microsoft.github.io/graphrag/../../prompt_tuning/auto_prompt_tuning/)

[Indexing Pipeline](https://microsoft.github.io/graphrag/../../index/overview/)

[Configuration](https://microsoft.github.io/graphrag/../overview/)

