# Services

This folder contains three different examples of RAG implemented using:
- Open AI API
- `Qwen2.5-3B-Instruct` model
- `Llama-3.2-1B-Instruct` model

All this implementations have been deployed online using [Modal](modal.com) and they're available at the following endpoints:
- `Llama`: https://ibbus93--nutritional-rag-service-llama-nutritionalragser-fc918b.modal.run
- `Qwen`: https://ibbus93--nutritional-rag-service-qwen-nutritionalragserv-7ae00e.modal.run
- `OpenAI`: https://ibbus93--nutritional-rag-service-openai-nutritionalragse-f9d7ea.modal.run  

In the following sections it will be explained how to setup the environment in order to both test and deploy the services using Modal.

## Before to start
Be sure that all the required dependencies are installed as [stated here](../README.md#how-to-run).

After the environment is ready:
- create an account on modal.com
- run the following command:
```bash
modal setup
```

The command will authenticate locally the user, so that it will be possible to run the service in the cloud. For any issue, please refer to the [official documentation](https://modal.com/docs/guide)

## Setup the environment
In order to run all the notebooks, it's needed to setup some secrets containing environment variables.
First of all, open the [secrets dashboard](https://modal.com/secrets), then create the following secrets with the required variables:

### mongodb-secret

| Variable             | Description                                 |
|----------------------|---------------------------------------------|
| MONGO_DB_USER        | MongoDB database username                   |
| MONGO_DB_PASSWORD    | MongoDB database password                   |
| MONGO_DB_CLUSTER_NAME| MongoDB cluster name                        |

### hf-secret

| Variable             | Description                                 |
|----------------------|---------------------------------------------|
| HF_TOKEN             | Hugging Face token for model access         |

## Testing locally

After all the setup done, it's possible to test locally a service using:

```bash
modal serve services/llama.py
```

Running this command will:
- create an ephemeral environment with the chosen service
- return an endpoint where to test the services deployed

Example of output of the command:

```bash
✓ Initialized. View run at https://modal.com/apps/username/main/ap-codes
✓ Created objects.
├── 🔨 Created mount /Users/user/rag-extract-calories/services/llama.py
├── 🔨 Created function NutritionalRagService.*.
└── 🔨 Created web endpoint for NutritionalRagService.get_nutritional_data => 
    https://username--nutritional-rag-service-llama-nutritionalra-ab123c-dev.modal.run
️️⚡️ Serving... hit Ctrl-C to stop!
└── Watching /Users/user/rag-extract-calories/services.
⠹ Running app...
```

This means that the application is running in watch mode, so it will be easy to test any change to the implementation. Once the terminal window is stopped, the ephemeral environment will be shut down.

## Deploy the service

Once the implementation is done, it's possible to deploy the service and keep it online indefinitely. In order to do this, run:

```bash
modal deploy services/llama.py
```

Just like the serve command, an endpoint for the deployed service will be returned.