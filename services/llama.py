import modal
from modal import App, Volume, Image
# Setup - define our infrastructure with code!

app = modal.App("nutritional-rag-service-llama")
secrets = [modal.Secret.from_name("hf-secret"), modal.Secret.from_name("mongodb-secret")]

image = Image.debian_slim().pip_install(
    "huggingface", "pymongo", "sentence_transformers", "transformers", "accelerate", "fastapi[standard]"
)

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("Loading model...")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, 
            cache_dir=CACHE_DIR,
            device_map="auto"    
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            cache_dir=CACHE_DIR, 
            device_map="auto"
        )

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
                    "limit": 3
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

    def get_json(self, text):
        import json
        import re

        # Find the starting position of the first JSON object
        start_of_json = text.find('{')
        
        if start_of_json != -1:
            # Use a regular expression to find the full JSON object
            # This pattern looks for a JSON-like structure enclosed in curly braces
            # with a closing brace `}` that isn't a part of an inner structure.
            # The `re.DOTALL` flag is used to match newlines as well.
            match = re.search(r'\{[^{}]*?\}', text[start_of_json:], re.DOTALL)
            
            if match:
                json_str = match.group(0)
                try:
                    # Parse the extracted string into a Python dictionary
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            else:
                print("No valid JSON structure found.")
        else:
            print("No JSON object found in the text.")

    @modal.fastapi_endpoint(method="POST")
    def get_nutritional_data(self, item: dict):
        import torch

        food = item['description']

        context = self.get_query_results(food)
        context_string = " - ".join([doc["text"] for doc in context])
        prompt = f"""Get the nutritional data of the following food ingredient: {food}. 
        Answer the question based only on the following context: {context_string}
        Reply only with a JSON that contains the following data: protein, carbohydrates, fats, calories, sugars, fibers. 
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation='do_not_truncate').to("cuda")
        attention_mask = torch.ones(input_ids.shape, device="cuda")
        outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100)
        decoded_output = self.tokenizer.decode(outputs[0]).replace('\n', '')

        return self.get_json(decoded_output)
