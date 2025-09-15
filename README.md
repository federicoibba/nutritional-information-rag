# Nutritional Information RAG

This project provides a Retrieval-Augmented Generation (RAG) service to extract nutritional information for food items using both OpenAI and open-source models, including Qwen and Llama. The workflow is organized in a series of Jupyter notebooks, modular Python services, and a final deployable API service.

## How to run
First of all, setup your environment installing all the needed dependencies. If you use Anaconda:
```bash
conda env create -f environment.yml
conda activate nutritional-rag-env
```

Using pip:
```bash
pip install -r requirements.txt
```

If you are interested on locally testing either the notebooks or the services, please refer to:
- [Notebooks documentation](./notebooks/README.md)
- [Services documentation](./services/README.md)

## Project Structure

- `datasets/food.csv` and `datasets/food_dataset.csv`: Raw and processed food nutrition datasets used as the data source for the project.
- `notebooks/`: the folder contains all the experiments and the vector database creations:
  - `0_dataset-food.ipynb`: Cleans and preprocesses the raw food dataset, extracting only the relevant columns (e.g., product name, fat, carbohydrates, proteins, calories, sugars, fiber).
  - `1_create_vectorstore.ipynb`: Creates a vector store from the cleaned dataset using sentence embeddings. This enables efficient semantic search for food items based on their descriptions.
  - `2.0_open-ai.ipynb`: Demonstrates the RAG pipeline using OpenAI models. It connects to MongoDB, retrieves relevant food entries, and generates nutritional information using OpenAI's API.
  - `2.1_open-source-llama.ipynb`: Uses the Llama open-source model for the RAG pipeline, showing how to generate nutritional information with this LLM.
  - `2.2_open-source-qwen.ipynb`: Uses the Qwen open-source model for the RAG pipeline, showing how to generate nutritional information with this LLM.
- `services/`: directory with all the services created: 
  - `llama.py`: Service logic for the Llama-based pipeline using `Llama-3.2-1B-Instruct`.
  - `qwen.py`: Service logic for the Qwen-based pipeline using `Qwen2.5-3B-Instruct`.
  - `openai-api.py`: Service logic for the OpenAI-based pipeline.

## How It Works

1. **Data Preparation**: The dataset is cleaned and relevant columns are selected in the first notebook.
2. **Vector Store Creation**: Food descriptions are embedded and stored in a vector database for semantic search.
3. **RAG Pipeline**: Given a food description, the system retrieves the most relevant entries from the vector store and uses a language model (OpenAI, Llama, or Qwen) to generate a structured nutritional profile.
4. **API Service**: The FastAPI-based service exposes a `/get_nutritional_data` endpoint (POST) that accepts a food description and returns nutritional information in JSON format. Both Llama and Qwen-based services are available.

## Dataset
The dataset used can be downloaded from Kaggle: [Food Nutrition Dataset](https://www.kaggle.com/datasets/shrutisaxena/food-nutrition-dataset)

## Deploy
The service is deployed online using [modal.com](https://modal.com/) service.

## Example API Usage
### Open source models
Send a POST request to the API endpoint with a JSON body:

```json
{
  "description": "cheddar cheese"
}
```

**API endpoints**: 
- Llama: https://ibbus93--nutritional-rag-service-llama-nutritionalragser-fc918b.modal.run
- Qwen: https://ibbus93--nutritional-rag-service-qwen-nutritionalragserv-7ae00e.modal.run

**Response**:
```json
{
  "protein": 22.5,
  "carbohydrates": 13.4,
  "fats": 0.8,
  "calories": 148,
  "sugars": 9.949999809,
  "fibers": 0.0
}
```

**Curl example**:
```bash
curl --location 'https://ibbus93--nutritional-rag-service-llama-nutritionalragser-fc918b.modal.run' \
--header 'Content-Type: application/json' \
--data '{
    "description": "Cheddar cheese"
}'
```

### Open AI 
The request is the same as the open source one, but an header `X-Open-AI-Api-Key` is required with the Open API key.

**API endpoint**: https://ibbus93--nutritional-rag-service-openai-nutritionalragse-f9d7ea.modal.run  

**Curl example**:

```bash
curl --location 'https://ibbus93--nutritional-rag-service-openai-nutritionalragse-f9d7ea.modal.run' \
--header 'Content-Type: application/json' \
--header 'X-Open-AI-Api-Key: sk-proj-your-api-key' \
--data '{
    "description": "Chicken"
}'
```