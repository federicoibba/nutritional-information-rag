# Notebooks

This folder contains all the notebooks used during the experimentation and the vector store creation.

## Before to start
Be sure that all the required dependencies are installed as [stated here](../README.md#how-to-run).

## Setup the environment
In order to run all the notebooks, it's needed to setup some environment variables:
1. Rename the [.env.example](../.env.example) file in `.env`
2. Substitute the mock variables with real values

Here is the list of the environment variables:

| Variable             | Description                                 |
|----------------------|---------------------------------------------|
| OPENAI_API_KEY       | OpenAI API key for accessing OpenAI models  |
| HF_TOKEN             | Hugging Face token for model access         |
| MONGO_DB_USER        | MongoDB database username                   |
| MONGO_DB_PASSWORD    | MongoDB database password                   |
| MONGO_DB_CLUSTER_NAME| MongoDB cluster name                        |

## Run the notebooks

After all the setup, just run in a terminal in the notebooks folder:

```bash
jupyter lab
```