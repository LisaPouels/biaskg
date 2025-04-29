from neo4j_graphrag.llm import LLMResponse, OllamaLLM, OpenAILLM, VertexAILLM
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever, Text2CypherRetriever, VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG
# from neo4j_graphrag.embeddings import OpenAIEmbeddings
# from langchain_ollama import ChatOllama, OllamaEmbeddings
from neo4j_graphrag.embeddings import OllamaEmbeddings
import os
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.retrievers.base import RetrieverResultItem
import neo4j
import pandas as pd
from dotenv import load_dotenv
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from evaluate_results import evaluate_results
from vertexai.generative_models import GenerationConfig
import google.generativeai as genai
from gemini_llm import GeminiLLM
from time import sleep

# Set the experiment name
mlflow.set_experiment("GraphRAG_Experiment2a_Retriever_K_value")

# Load environment variables from .env file
load_dotenv(override=True)

# Set the index name
INDEX_NAME = "startIndex"

# 1. Knowledge Graph connection
# Load environment variables
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) # connect to the database

# 2. Embedding
embedder = OllamaEmbeddings(model="nomic-embed-text") # set the embedding model

# 3. Retriever
def result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """
    Format the result from the database query into a RetrieverResultItem.
    The retrieved start nodes are connected to an edge (relationship) and an end node.
    The retrieved end nodes are connected to the start node with an edge (relationship).

    Args:
        record (neo4j.Record): The record returned from the database query.
    Returns:
        RetrieverResultItem: The formatted result item. This includes the content and metadata.
    The content is a string that contains the start node, the relationship, and the end node.
    The metadata includes the start node and the score.
    """
    content=""
    for i in range(len(record.get('top_triplets'))):
        content += f"{record.get('top_triplets')[i].get('subject')} {record.get('top_triplets')[i].get('relationship')} {record.get('top_triplets')[i].get('object')},"
    return RetrieverResultItem(
        content=content,
        metadata={
            "startNode": record.get('node'),
            "score": record.get("score"),
        }
    )

# define the retrieval query
retrieval_query = """
    // Step 1: Find neighbors of the retrieved node
    MATCH (node)-[r:RELATIONSHIP]->(e:EndNode)

    // Step 2: Compute cosine similarity manually between input and e
    WITH node, r, e,
        gds.similarity.cosine(node.embedding, e.embedding) AS e_similarity,
        score AS node_similarity // manually preserve the score for 'node'

    // Step 3: Top-k neighbors based on similarity
    ORDER BY e_similarity DESC
    WITH node, node_similarity, COLLECT(DISTINCT {entity: e, sim: e_similarity})[0..$top_k] AS top_e

    // Step 4: Combine node + top_e into one list
    WITH node, node_similarity,
        [{entity: node, sim: node_similarity}] + top_e AS nodes

    UNWIND nodes AS entity_info
    WITH node, entity_info.entity AS n, entity_info.sim AS similarity

    // Step 5: Get all outgoing edges for all relevant nodes
    MATCH (n)-[r1:RELATIONSHIP]->(e1:EndNode)

    // Step 6: Collect and rank triplets
    WITH node, n, r1, e1, similarity
    ORDER BY similarity DESC
    WITH node,
        COLLECT({subject: n.text, relationship: r1.text, object: e1.text}) AS triplets,
        AVG(similarity) AS avg_similarity

    // Step 7: Return
    RETURN
        node.text AS node,
        avg_similarity AS score,
        triplets[0..$top_k] AS top_triplets
"""
# initialize the retriever
retriever = VectorCypherRetriever(
    driver=driver,
    index_name=INDEX_NAME,
    retrieval_query=retrieval_query,
    embedder=embedder,
    result_formatter=result_formatter,
)

# 4. Load the user prompts
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

# models = ["mistral", "llama3.2", "qwen2.5", "deepseek-v2", "falcon", "gpt-4.1-nano", "gemini-2.0-flash"] #all models
models = ["mistral", "llama3.2", "qwen2.5", "deepseek-v2", "falcon", "gpt-4.1-nano"] #all models except gemini
# models = ["mistral", "llama3.2", "qwen2.5", "falcon", "deepseek-v2"] # just the ollama models
# models = ["deepseek-v2"]
sleep_time = 0
k_values = [1,3,5,10] # values tested in the biasKG paper, except for 0 which is not possible
# k_values = [5] # default, from biasKG paper
timestamp = pd.Timestamp.now().strftime("%m%d_%H%M") # set the timestamp for the experiment

# 5. Loop through the models and k values
for model in models:
    sample_size = len(df_prompts)
    # Loop through the k values
    for k in k_values:
        print(f"Running experiments for LLM={model} with k={k}")

        # Initialize the LLM
        if model == "gemini-2.0-flash":
            sleep(sleep_time) # sleep to avoid rate limiting for Gemini
            sleep_time = 60
            mlflow.gemini.autolog()
            llm = GeminiLLM(model_name=model)
            sample_size = 15  # select only the first 15 prompts for the gemini model
        elif model == "gpt-4.1-nano":
            sleep_time = 0
            mlflow.openai.autolog()
            llm = OpenAILLM(
                model_name=model,
                model_params={"temperature":0}
            )
        else:
            sleep_time = 0
            mlflow.openai.autolog()
            # Using OpenAI LLM ensures mlflow can track the traces. It still runs locally.
            llm = OpenAILLM(
                model_name=model,
                model_params={"temperature": 0, "max_tokens": 1000}, # max_tokens prevents the model from generating too much text and never finishing (falcon model had that issue)
                base_url=os.getenv("BASE_URL_OLLAMA"),
            )
        # Create a dataframe to store the answers
        df_answers = pd.DataFrame(columns=['context', 'question', 'ans0', 'ans1', 'ans2', 'label', 'RAG_Answer', 'context_condition', 'question_polarity', 'category', 'target_loc', 'retriever_result'])	

        # Initialize the RAG pipeline
        rag = GraphRAG(retriever=retriever, llm=llm)

        # Select the prompts
        prompts = df_prompts.iloc[:sample_size]

        with mlflow.start_run(run_name=f"{model}_k{k}_{timestamp}_bbq_experiment"):
            mlflow.log_param("model", model)
            mlflow.log_param("retriever", "VectorCypherRetriever")
            mlflow.log_param("embedder model", "nomic-embed-text")
            mlflow.log_param("retrieval query", retrieval_query)
            mlflow.log_param("sample size", len(prompts))
            mlflow.log_param("k", k)
            mlflow.log_input(dataset)

            for i in range(len(prompts)):
                question = prompts.iloc[i]['question']
                context = prompts.iloc[i]['context']
                answer_options =  prompts.iloc[i]['ans0'],prompts.iloc[i]['ans1'],prompts.iloc[i]['ans2']
                # query_text = f"{context} {question} Answer with one of the following options: {answer_options}"
                query_text = f"{context} {question} Answer with one of the following options: {answer_options}. Start with the correct answer option so it can be checked, optionally follow it with an explanation."

                response = rag.search(query_text=query_text, retriever_config={"top_k": k, "query_params": {"k": k}}, return_context=True)
                # add response to df_answers together with the context and question
                df_answers = pd.concat([df_answers, pd.DataFrame({'context': context, 'question': question, 'ans0': prompts.iloc[i]['ans0'], 'ans1': prompts.iloc[i]['ans1'], 'ans2': prompts.iloc[i]['ans2'], 'label': prompts.iloc[i]['label'], 'RAG_Answer': response.answer, 'context_condition': prompts.iloc[i]['context_condition'], 'question_polarity': prompts.iloc[i]['question_polarity'], 'category': prompts.iloc[i]['category'], 'target_loc': prompts.iloc[i]['target_loc'], 'retriever_result': [response.retriever_result.items]}, index=[0])], ignore_index=True)

            # print(df_answers.head(10))
            #evaluate the results
            overall_accuracy, accuracy_ambiguous, accuracy_disambiguated, accuracy_cost_bias_nonalignment, bias_disambig, bias_ambig = evaluate_results(df_answers)
            # add the results to the dataframe
            df_answers['Accuracy'] = overall_accuracy
            df_answers['Accuracy_ambiguous'] = accuracy_ambiguous
            df_answers['Accuracy_disambiguous'] = accuracy_disambiguated
            df_answers['Accuracy_cost_bias_nonalignment'] = accuracy_cost_bias_nonalignment
            df_answers['Bias_disambig'] = bias_disambig 
            df_answers['Bias_ambig'] = bias_ambig
            # log the results
            mlflow.log_metric("overall_accuracy", overall_accuracy)
            mlflow.log_metric("accuracy_ambiguous", accuracy_ambiguous)
            mlflow.log_metric("accuracy_disambiguous", accuracy_disambiguated)
            mlflow.log_metric("accuracy_cost_bias_nonalignment", accuracy_cost_bias_nonalignment)
            mlflow.log_metric("bias_disambig", bias_disambig)
            mlflow.log_metric("bias_ambig", bias_ambig)

            #save the dataframe to a csv file, remove enters from the text
            df_answers['RAG_Answer'] = df_answers['RAG_Answer'].str.replace('\n', ' ')
            df_answers.to_csv(f"Experiments/2_Retriever/2a_k_value/{model}_k{k}_{timestamp}_bbq_experiment.csv", index=False)