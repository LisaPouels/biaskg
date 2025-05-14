from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.embeddings import OllamaEmbeddings
import os
import pandas as pd
from dotenv import load_dotenv
import mlflow
from components.retriever import result_formatter, RETRIEVAL_QUERY_SIMILARITY, RETRIEVAL_QUERY_PAGERANK
from components.runner import run_experiment
from components.reranker import RerankableRetriever
import logging

logger = logging.getLogger("httpx")
logger.setLevel(logging.WARNING)

# Set the experiment name
mlflow.set_experiment("GraphRAG_Experiment")

# Load environment variables from .env file
load_dotenv(override=True)

# Set the index name
INDEX_NAME = "startIndex"

# 1. Knowledge Graph connection
# Load environment variables
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password), notifications_min_severity="OFF") # connect to the database

# 2. Load the user prompts
data_path = os.getenv("DATA_PATH")
n_prompts = os.getenv("N_PROMPTS") # number of prompts to sample

df_bbq = pd.read_csv(data_path)
if n_prompts == 'None' or n_prompts == None or n_prompts == "":
    print(f"No number of prompts specified. Using all prompts.")
    n_prompts = len(df_bbq) # if no number of prompts is specified, use all prompts
if int(n_prompts) > len(df_bbq):
    print(f"Number of prompts specified ({n_prompts}) is greater than the number of prompts in the dataset ({len(df_bbq)}). Using all prompts.")
    n_prompts = len(df_bbq) # if no number of prompts is specified, use all prompts

print(f"Sampling {n_prompts} prompts from {data_path}")
df_prompts = df_bbq.sample(int(n_prompts), random_state=42).reset_index(drop=True)  #sample prompts, random_state=42 for reproducibility

dataset = mlflow.data.from_pandas(df_prompts, name="bbq_sample")

# 3. Embedding
embedder = OllamaEmbeddings(model="nomic-embed-text") # set the embedding model

# 4. Retriever
retrievers = [
    ("Original", RETRIEVAL_QUERY_SIMILARITY, "Original"),
    ("Original", RETRIEVAL_QUERY_PAGERANK, "Pagerank"),
    ("Reranker", RETRIEVAL_QUERY_SIMILARITY, "Original"),
    ("Reranker", RETRIEVAL_QUERY_PAGERANK, "Pagerank"),
]

# 5. Generation
# models = ["mistral", "llama3.2", "qwen2.5", "deepseek-v2", "falcon", "gpt-4.1-nano", "gemini-2.0-flash"] #all models
# models = ["mistral", "llama3.2", "qwen2.5", "deepseek-v2", "falcon", "gpt-4.1-nano"] #all models except gemini
# models = ["mistral", "llama3.2", "qwen2.5", "falcon", "deepseek-v2"] # just the ollama models
models = ["qwen2.5"]
sleep_time = 0
# k_values = [1,3,5,10] # values tested in the biasKG paper, except for 0 which is not possible
k_values = [5] # default, from biasKG paper
timestamp = pd.Timestamp.now().strftime("%m%d_%H%M") # set the timestamp for the experiment

# Loop through the retrievers and queries
for retriever_name, retrieval_query, retriever_type in retrievers:
    if retriever_name == "Original":
        retriever = VectorCypherRetriever(
            driver=driver,
            index_name=INDEX_NAME,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=result_formatter,
        )
    elif retriever_name == "Reranker":
        retriever = RerankableRetriever(
            driver=driver,
            index_name=INDEX_NAME,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=result_formatter,
        )
    else:
        raise ValueError(f"Unknown retriever name: {retriever_name}")
    # Loop through the models and k values
    for model in models:
        for k in k_values:
            print(f"Running experiments for LLM={model} with k={k}, retriever={retriever_name}, retriever type={retriever_type}")
            run_experiment(model, k, df_prompts, retriever, timestamp, dataset, retrieval_query, retriever_name, retriever_type)