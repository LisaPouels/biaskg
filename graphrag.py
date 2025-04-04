from neo4j_graphrag.llm import LLMResponse, OllamaLLM, OpenAILLM
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

# Set the experiment name
mlflow.set_experiment("GraphRAG_Experiment")
mlflow.openai.autolog()

# Load environment variables from .env file
load_dotenv(override=True)

# 1. Neo4j driver
# URI = "neo4j://localhost:7687"
# AUTH = ("neo4j", "password")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
model = os.getenv("MODEL")
print(f"Using model: {model}")

INDEX_NAME = "startIndex"

# Connect to Neo4j database
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# 2. Retriever
# Create Embedder object, needed to convert the user question (text) to a vector
embedder = OllamaEmbeddings(model="nomic-embed-text")

# 3. LLM
llm = OllamaLLM(
    model_name=model,
)
# llm = OpenAILLM(
#     model_name=model,
#     model_params={"temperature": 0},
#     base_url="http://localhost:11434/v1",
# )



# Initialize the retriever
# retriever = VectorRetriever(driver, INDEX_NAME, embedder)
# retriever = Text2CypherRetriever(driver, llm=llm)
def result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    content=""
    for i in range(len(record.get('node_rel'))):
        content += f"{record.get('node')} {record.get('node_rel')[i]} {record.get('e')[i]},"
    for i in range(len(record.get('rel_node'))):
        content += f"{record.get('s')[i]} {record.get('rel_node')[i]} {record.get('node')},"
    return RetrieverResultItem(
        content=content,
        metadata={
            "startNode": record.get('node'),
            "score": record.get("score"),
        }
    )

retrieval_query = """
    RETURN 
        node.text AS node, 
        score,
        COLLECT {MATCH (node)-[r:RELATIONSHIP]-(e:EndNode) RETURN r.text} AS node_rel,
        COLLECT {MATCH (node)-[r:RELATIONSHIP]-(e:EndNode) RETURN e.text} AS e,
        COLLECT {MATCH (s:StartNode)-[r:RELATIONSHIP]-(node) RETURN s.text} AS s,
        COLLECT {MATCH (s:StartNode)-[r:RELATIONSHIP]-(node) RETURN r.text} AS rel_node
"""
retriever = VectorCypherRetriever(
    driver=driver,
    index_name=INDEX_NAME,
    retrieval_query=retrieval_query,
    embedder=embedder,
    result_formatter=result_formatter,
)

# Initialize the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)
with mlflow.start_run():
    # Query using bbq data
    df_bbq = pd.read_csv("Data/bbq_sample.csv")
    #sample data
    df_prompts = df_bbq.sample(20, random_state=42).reset_index(drop=True)
    # df_prompts['RAG_Answer'] = None
    df_answers = pd.DataFrame(columns=['context', 'question', 'ans0', 'ans1', 'ans2', 'label', 'RAG_Answer', 'context_condition', 'question_polarity', 'category', 'retriever_result'])	
    # rag_answers = []
    for i in range(len(df_prompts)):
        question = df_prompts.iloc[i]['question']
        context = df_prompts.iloc[i]['context']
        answer_options =  df_prompts.iloc[i]['ans0'],df_prompts.iloc[i]['ans1'],df_prompts.iloc[i]['ans2']
        query_text = f"{context} {question} Answer with one of the following options: {answer_options}"

        response = rag.search(query_text=query_text, retriever_config={"top_k": 3}, return_context=True)
        # add response to df_answers together with the context and question
        df_answers = pd.concat([df_answers, pd.DataFrame({'context': context, 'question': question, 'ans0': df_prompts.iloc[i]['ans0'], 'ans1': df_prompts.iloc[i]['ans1'], 'ans2': df_prompts.iloc[i]['ans2'], 'label': df_prompts.iloc[i]['label'], 'RAG_Answer': response.answer, 'context_condition': df_prompts.iloc[i]['context_condition'], 'question_polarity': df_prompts.iloc[i]['question_polarity'], 'category': df_prompts.iloc[i]['category'], 'retriever_result': [response.retriever_result.items]}, index=[0])], ignore_index=True)

    # df_prompts['RAG_Answer'] = rag_answers
    # print(df_prompts[['question', 'context', 'RAG_Answer', 'context_condition']].head(10))
    print(df_answers.head(10))

    #save the dataframe to a csv file, remove enters from the text
    df_answers['RAG_Answer'] = df_answers['RAG_Answer'].str.replace('\n', ' ')
    timestamp = pd.Timestamp.now().strftime("%m%d_%H%M")
    df_answers.to_csv(f"Experiments/{model}_{timestamp}_bbq_experiment.csv", index=False)