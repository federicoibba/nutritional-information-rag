import modal
from modal import App, Volume, Image
from fastapi import Request, HTTPException
from pydantic import BaseModel
# Setup - define our infrastructure with code!

app = modal.App("nutritional-rag-service-openai")
secrets = [modal.Secret.from_name("hf-secret"), modal.Secret.from_name("mongodb-secret")]

image = Image.debian_slim().pip_install(
    "huggingface", "pymongo", "sentence_transformers", "transformers", "accelerate", "fastapi[standard]", "torch", "lm-format-enforcer", "optimum", "openai"
)

# Constants
GPU = "T4"
BASE_MODEL = "gpt-4o-mini"
SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

class AnswerFormat(BaseModel):
    protein: float
    carbohydrates: float
    fats: float
    calories: float
    sugar: float
    fiber: float

@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets, 
    gpu=GPU, 
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class NutritionalRagService:
    def connect_to_mongo(self):
        import os
        from pymongo import MongoClient
        
        # Setup MongoDB Connection
        MONGO_DB_USER = os.environ.get('MONGO_DB_USER')
        MONGO_DB_PASSWORD = os.environ.get('MONGO_DB_PASSWORD')
        MONGO_DB_CLUSTER_NAME = os.environ.get('MONGO_DB_CLUSTER_NAME')

        DB_NAME = 'nutritional_rag'
        COLLECTION_NAME = 'food'

        print(f"""Connecting to MongoDB {MONGO_DB_CLUSTER_NAME}...""")


        uri = f"mongodb+srv://{MONGO_DB_USER}:{MONGO_DB_PASSWORD}@{MONGO_DB_CLUSTER_NAME}.i1ndjzi.mongodb.net/?retryWrites=true&w=majority&appName={MONGO_DB_CLUSTER_NAME}"

        client = MongoClient(uri)
        self.collection = client[DB_NAME][COLLECTION_NAME]

        print("Connected to MongoDB")

    @modal.enter()
    def setup(self):
        from sentence_transformers import SentenceTransformer

        self.connect_to_mongo()
        self.transformer = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)


    def get_query_results(self, query):
        """Gets results from a vector search query."""
        array_of_results = []

        query_embedding = self.transformer.encode(query).tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "food_vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "exact": True,
                    "limit": 5
                }
            }, 
            {
                "$project": {
                    "_id": 0,
                    "text": 1
                }
            }
        ]

        results = self.collection.aggregate(pipeline)

        for doc in results:
            array_of_results.append(doc)

        return array_of_results

    @modal.fastapi_endpoint(method="POST")
    async def get_nutritional_data(self, request: Request, item: dict):
        from openai import OpenAI
        import json

        # Get OpenAI API key from headers
        if 'X-Open-AI-Api-Key' not in request.headers:
            raise HTTPException(status_code=400, detail="Missing OpenAI API key in 'X-Open-AI-Api-Key' header")

        openai_api_key = request.headers['X-Open-AI-Api-Key']

        food = item['description']

        # Create prompt starting from food and extracting context from MongoDB
        context = self.get_query_results(food)
        context_string = " - ".join([doc["text"] for doc in context])
        prompt = f"""
        Get the nutritional data of the following food ingredient: {food}
        The food ingredient data is intended to be for 100 grams of it. If you can't find the ingredient, return the values for it as raw/not cooked.

        Answer the question based only on the following context: {context_string}

        Reply only with a JSON that contains the following data: protein, carbohydrates, fats, calories, sugars, fibers. 
        Please be strict to this JSON format, if you don't know the answer use the context as much as possible.
        """

        openai_client = OpenAI(api_key=openai_api_key)

        completion = openai_client.chat.completions.create(
        model=BASE_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])

        response = completion.choices[0].message.content
        response = response.replace('```json', '').replace('```', '').replace('\n', '')

        return json.loads(response)
