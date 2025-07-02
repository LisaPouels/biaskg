# BiasKG: Evaluating Bias and Robustness in Graph RAG Systems

This repository provides a experimental framework for evaluating the impact of various components within a Retrieval-Augmented Generation (RAG) pipeline built on a Knowledge Graph (KG). The primary focus is to measure and understand how choices in retrieval strategy, large language models (LLMs), and prompt quality contribute to social biases and affect the overall robustness of the system.

This repository expands upon the [BiasKG repository](https://github.com/VectorInstitute/biaskg), and uses their knowledge graph and part of the implementation presented in the corresponding [paper](https://arxiv.org/pdf/2405.04756).

The evaluation is performed using the [Bias Benchmark for QA (BBQ) dataset](https://github.com/nyu-mll/BBQ) to systematically assess model outputs for accuracy and bias. The framework uses `mlflow` for meticulous experiment tracking, allowing for easy comparison of different configurations.

## Key Features

*   **Modular Graph RAG Pipeline:** Built using the powerful `neo4j-graphrag` library to connect a Neo4j knowledge graph with various LLMs.
*   **Multi-LLM Support:** Systematically evaluate and compare a wide range of LLMs, including local models via Ollama (Llama3.2, Mistral, Qwen2.5), and API-based models (OpenAI's GPT series, Google's Gemini).
*   **Advanced Retrieval Strategies:** Compare different retrieval methods, including standard vector similarity search, and a PageRank-enhanced approach to leverage graph topology.
*   **Reranking Integration:** Includes an optional reranking step using `flashrank` to refine retrieval results before passing them to the LLM.
*   **Prompt Robustness Testing:** A powerful prompt perturbation engine to test the system's resilience against various input degradations, from simple typos to complex paraphrasing and style changes.
*   **Comprehensive Bias & Accuracy Evaluation:** Implements the evaluation methodology from the BBQ paper to calculate overall accuracy, performance on ambiguous vs. disambiguated questions, and specific bias scores.
*   **End-to-End Experiment Tracking:** Leverages `mlflow` to log all parameters, inputs, outputs, and evaluation metrics, providing a full audit trail for each experiment.

## Repository Structure

```
.
├── main.py                                             # Main entry point to configure and run experiments
├── requirements.txt                                    # Python dependencies
├── .env.template                                        # Example environment file
│
├── components/
│   ├── runner.py                                       # Handles the logic for a single experiment run
│   ├── retriever.py                                    # Defines retrieval queries and result formatters
│   ├── reranker.py                                     # Implements the RerankableRetriever with flashrank
│   ├── generation.py                                   # Manages LLM initialization and prompt construction/perturbation
│   ├── evaluate_results.py                             # Logic for calculating bias and accuracy metrics from BBQ dataset
│   ├── gemini_llm.py                                   # Custom LLM interface for Google Gemini
│   └── prompt_perturbation/
│       ├── character.py                                # Character-level text perturbations
│       ├── word.py                                     # Word-level text perturbations
│       ├── sentence.py                                 # Sentence-level text perturbations (paraphrasing, style transfer)
│       ├── styleformer.py                              # Helper for style transfer
│       └── parrot.py                                   # Helper for paraphrasing
│
├── components/
│   ├── analyze_bbq_data.ipynb                          # Creates sample from the original BBQ data
│   └── knowledge_graph_to_neo4j.ipynb                  # Creates knowledge graph in Neo4j with the data from BiasKG
│
├── dynamic_kg_generator/                               # Part of the original repository
│
├── kg_benchmark/                                       # Part of the original repository
│
└── Experiments/
    ├── 1_LLM/                                          # Directory to save CSV results of experiment 1
    ├── 2_Retriever/                                    # Directory to save CSV results of experiment 2
    ├── 3_Prompts/                                      # Directory to save CSV results of experiment 3
    └── experiment1_LLM_evaluation.ipynb                # Notebooks with quantitative and qualitative analyses per experiment
    └── experiment2a_k_retriever_evaluation.ipynb       # Notebooks with quantitative and qualitative analyses per experiment
    └── experiment2b_retriever_evaluation.ipynb         # Notebooks with quantitative and qualitative analyses per experiment
    └── experiment3_prompts_evaluation.ipynb            # Notebooks with quantitative and qualitative analyses per experiment
    └── experiments_qualitative_evaluation.ipynb        # Notebooks with quantitative and qualitative analyses per experiment
```

## Setup and Installation

### 1. Prerequisites

*   Python 3.10+
*   Git
*   [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose

### 2. Clone Repository

```bash
git clone https://github.com/LisaPouels/biaskg.git
cd biaskg
```

### 3. Set Up Python Environment

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The first time you run a perturbation that uses `nltk` (e.g., synonym replacement), it will automatically download the required packages (`stopwords`, `punkt`, `wordnet`).

### 4. Set Up Services (Neo4j & Ollama)

This project relies on a Neo4j database and the Ollama service for running local LLMs. You can use Docker to run them.

**A. Neo4j Database**

1.  Make sure you have Docker installed and running.
2.  Run a Neo4j instance. Ensure you include the **APOC** and **Graph Data Science (GDS)** plugins, as GDS is required for the PageRank retriever.

    ```bash
    docker run \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password \
        -e NEO4J_PLUGINS='["apoc", "gds"]' \
        --name neo4j-biaskg \
        neo4j:5.20-enterprise
    ```

**B. Ollama Service**

1.  Follow the official instructions to [install and run Ollama](https://ollama.com/).
2.  Pull the required models specified in `main.py`. This includes the embedding model and the generation models.

    ```bash
    # Embedding model
    ollama pull nomic-embed-text

    # Example generation models
    ollama pull llama3.2
    ollama pull mistral
    ollama pull qwen2.5
    ollama pull deepseek-v2
    ollama pull falcon
    ```

### 5. Load Data into Neo4j

This framework assumes your Neo4j database is already populated with a knowledge graph. The nodes in the graph must have `text` and `embedding` properties for the retriever to function.

1.  **Ingest Data:** Use a tool like `neo4j-graphrag` or a custom script to ingest your source documents and create a graph. The retrieval queries in `components/retriever.py` expect nodes with labels like `StartNode` and `EndNode` connected by `RELATIONSHIP` relationships.

2.  **Create Vector Index:** You must create a vector index in Neo4j for vector similarity search. You can do this from the Neo4j Browser. The index name should match the `INDEX_NAME` variable in `main.py` (`startIndex` by default).

    ```cypher
    -- Replace 'StartNode' with the label of your main nodes
    -- The dimension 768 is for nomic-embed-text
    CREATE VECTOR INDEX startIndex IF NOT EXISTS
    FOR (n:StartNode) ON (n.embedding)
    OPTIONS { indexConfig: {
     `vector.dimensions`: 768,
     `vector.similarity_function`: 'cosine'
    }}
    ```

### 6. Configure Environment Variables

1.  Copy the example environment file.
    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file with your specific configurations.

    ```dotenv
    # Neo4j Credentials
    NEO4J_URI="bolt://localhost:7687"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="password"

    # Ollama Base URL
    BASE_URL_OLLAMA="http://localhost:11434/v1"

    # Path to the BBQ dataset CSV file
    DATA_PATH="path/to/your/bbq.csv"
    # Number of prompts to sample from the dataset. Use "" or "None" to use all.
    N_PROMPTS="100"

    # API Keys for external services (if used)
    OPENAI_API_KEY="your-openai-key"
    GENAI_API_KEY="your-google-gemini-key"
    ```

## Running Experiments

### 1. Configure the Experiment

Open `main.py` to configure the experiment runs. You can control which components are tested by modifying these lists:

*   `retrievers`: A list of tuples defining the retriever configurations. Each tuple contains `(RetrieverName, RetrievalQuery, RetrieverType)`.
    *   `RetrieverName`: "Original" or "Reranker".
    *   `RetrievalQuery`: `RETRIEVAL_QUERY_SIMILARITY` or `RETRIEVAL_QUERY_PAGERANK`.
    *   `RetrieverType`: "Original" or "Pagerank" (used for logging and setup).
*   `perturbation_list`: A list of strings specifying which prompt perturbations to apply. "original" runs without any perturbation.
*   `models`: A list of LLM model names to test.
*   `k_values`: A list of integers for the `k` parameter (number of retrieved contexts).

### 2. Start MLflow Tracking UI

Before running the experiment, you can start the MLflow UI in a separate terminal to monitor the runs in real-time.

```bash
mlflow ui
```

Navigate to `http://localhost:5000` in your browser. The experiment will be logged under the name set in `main.py` (e.g., `GraphRAG_Experiment3_Prompts`).

### 3. Run the Main Script

Execute `main.py` from your terminal to start the experiment loop.

```bash
python main.py
```

The script will iterate through all combinations of retrievers, perturbations, models, and k-values you configured. For each combination, it will:
1.  Process the prompts from the BBQ dataset.
2.  Run the Graph RAG pipeline.
3.  Evaluate the results for accuracy and bias.
4.  Log all parameters, metrics, and results to MLflow.
5.  Save a detailed CSV of the run's outputs in the `Experiments/3_Prompts/` directory.

## Evaluation Metrics

The evaluation logic in `evaluate_results.py` calculates several key metrics based on the BBQ dataset's structure:

*   **Overall Accuracy:** The percentage of correctly answered questions across the entire sample.
*   **Accuracy Ambiguous:** Accuracy on questions where the context is ambiguous.
*   **Accuracy Disambiguous:** Accuracy on questions where the context is not ambiguous.
*   **Bias Disambig:** A score indicating the model's bias on non-ambiguous questions. It measures the tendency to choose the stereotyped answer.
*   **Bias Ambig:** A score indicating the model's bias on ambiguous questions, where it must rely on its internal knowledge/biases.