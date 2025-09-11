# Nutritional Information RAG

This project provides a Retrieval-Augmented Generation (RAG) service to extract nutritional information for food items using both OpenAI and open-source models. The workflow is organized in a series of Jupyter notebooks and a final deployable API service.

## Project Structure

- `0_dataset-food.ipynb`: Cleans and preprocesses the raw food dataset, extracting only the relevant columns (e.g., product name, fat, carbohydrates, proteins, calories, sugars).
- `1_create_vectorstore.ipynb`: Creates a vector store from the cleaned dataset using sentence embeddings. This enables efficient semantic search for food items based on their descriptions.
- `2_open-ai.ipynb`: Demonstrates the RAG pipeline using OpenAI models. It connects to MongoDB, retrieves relevant food entries, and generates nutritional information using OpenAI's API.
- `2.1_open-source.ipynb`: Similar to the previous notebook, but uses an open-source language model for the RAG pipeline instead of OpenAI.
- `nutritional_rag_service.py`: Implements the final API service using FastAPI and Modal. The service exposes an endpoint to receive a food description and returns its nutritional data (protein, carbohydrates, fats, calories, sugars, fibers) as JSON. It leverages a vector search over the MongoDB collection and generates answers using a local LLM.

## How It Works

1. **Data Preparation**: The dataset is cleaned and relevant columns are selected.
2. **Vector Store Creation**: Food descriptions are embedded and stored in a vector database for semantic search.
3. **RAG Pipeline**: Given a food description, the system retrieves the most relevant entries from the vector store and uses a language model to generate a structured nutritional profile.
4. **API Service**: The FastAPI-based service exposes a `/get_nutritional_data` endpoint (POST) that accepts a food description and returns nutritional information in JSON format.

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