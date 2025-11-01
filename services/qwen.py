import modal
from modal import App, Volume, Image
from pydantic import BaseModel
# Setup - define our infrastructure with code!

app = modal.App("nutritional-rag-service-qwen")
secrets = [modal.Secret.from_name("hf-secret"), modal.Secret.from_name("mongodb-secret")]

image = Image.debian_slim().pip_install(
    "huggingface", "pymongo", "sentence_transformers", "transformers", "accelerate", "fastapi[standard]", "torch", "lm-format-enforcer", "optimum", "outlines", "bitsandbytes"
)
## Modal settings
GPU = "T4"
CACHE_DIR = "/cache"
# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0

# Application Constants
BASE_MODEL = "qwen/Qwen2.5-7B-Instruct"
SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
FOOD_DB_ITEMS = 3


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
    
    def init_model(self):
        from outlines import Generator, from_transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print("Loading model...")

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        quantized_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=quantization_config)

        model = from_transformers(
            quantized_model,
            AutoTokenizer.from_pretrained(BASE_MODEL),
        )

        self.generator = Generator(model, AnswerFormat)

        print(f"""Model {BASE_MODEL} loaded""")

    @modal.enter()
    def setup(self):
        from sentence_transformers import SentenceTransformer

        self.init_model()
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
                    "limit": FOOD_DB_ITEMS
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
    def get_nutritional_data(self, item: dict):
        import json

        food = item['description']

        context = self.get_query_results(food)
        context_string = " \n ".join([doc["text"] for doc in context])
        prompt = f"""
            Please use only the following context to answer the question.
            **Precedence Rule: Always choose the nutritional data for RAW foods if available.**

            Get the nutritional data of the following food ingredient: **{food}**.
            CONTEXT OPTIONS:
            {context_string}
        """

        print("Generated prompt:", prompt)

        # Extract the results
        return json.loads(self.generator(
            prompt,
            max_new_tokens=200,
        ))
