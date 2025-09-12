# Nutritional Information RAG

This project provides a Retrieval-Augmented Generation (RAG) service to extract nutritional information for food items using both OpenAI and open-source models, including Qwen. The workflow is organized in a series of Jupyter notebooks and a final deployable API service.

## Project Structure

- `datasets/food.csv` and `datasets/food_dataset.csv`: Raw and processed food nutrition datasets used as the data source for the project.
- `notebooks/0_dataset-food.ipynb`: Cleans and preprocesses the raw food dataset, extracting only the relevant columns (e.g., product name, fat, carbohydrates, proteins, calories, sugars).
- `notebooks/1_create_vectorstore.ipynb`: Creates a vector store from the cleaned dataset using sentence embeddings. This enables efficient semantic search for food items based on their descriptions.
- `notebooks/2.0_open-ai.ipynb`: Demonstrates the RAG pipeline using OpenAI models. It connects to MongoDB, retrieves relevant food entries, and generates nutritional information using OpenAI's API.
- `notebooks/2.1_open-source.ipynb`: Uses an open-source language model for the RAG pipeline instead of OpenAI.
- `notebooks/2.2_open-source-qwen.ipynb`: Uses the Qwen open-source model for the RAG pipeline, showing how to generate nutritional information with this LLM.
- `nutritional_rag_service.py`: Implements the final API service using Llama model, FastAPI and Modal. This service exposes an endpoint to receive a food description and returns its nutritional data (protein, carbohydrates, fats, calories, sugars, fibers) as JSON, using [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)  for answer generation.
- `nutritional_rag_service-qwen.py`: Implements the API service using the Qwen model, FastAPI, and Modal. This service exposes an endpoint to receive a food description and returns its nutritional data (protein, carbohydrates, fats, calories, sugars, fibers) as JSON, using [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) for answer generation.

## How It Works

1. **Data Preparation**: The dataset is cleaned and relevant columns are selected in the first notebook.
2. **Vector Store Creation**: Food descriptions are embedded and stored in a vector database for semantic search.
3. **RAG Pipeline**: Given a food description, the system retrieves the most relevant entries from the vector store and uses a language model (OpenAI, open-source, or Qwen) to generate a structured nutritional profile.
4. **API Service**: The FastAPI-based service exposes a `/get_nutritional_data` endpoint (POST) that accepts a food description and returns nutritional information in JSON format. Both Llama and Qwen-based services are available.

## Dataset
The dataset used can be downloaded from Kaggle: [Food Nutrition Dataset](https://www.kaggle.com/datasets/shrutisaxena/food-nutrition-dataset)

## Deploy
The service is deployed online using [modal.com](https://modal.com/) service.

## Example API Usage
API endpoint: https://ibbus93--nutritional-rag-service-nutritionalragservice-g-98e082.modal.run

Send a POST request to the API endpoint with a JSON body:

```json
{
  "description": "cheddar cheese"
}
```

Response:
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

Curl example:
```bash
curl --location 'https://ibbus93--nutritional-rag-service-nutritionalragservice-g-98e082.modal.run' \
--header 'Content-Type: application/json' \
--data '{
    "description": "Cheddar cheese"
}'
```