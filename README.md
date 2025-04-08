# BiasKG

## Original paper and repository
**BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models** \
[Original repository](https://github.com/VectorInstitute/biaskg/tree/main)\
[paper link here](https://arxiv.org/pdf/2405.04756)


## File structure
- Data: includes the BBQ data files sorted per category, the BiasKG data, and a sample of the BBQ data that consists of multiple categories.
- Experiments: results from experiments, including the prompt, response, accuracy, fairness and more details
- dynamic_kg_generator: from the original repository, includes code to create the knowledge graph and retrieve nodes
- kg_benchmark: from the original repository, includes code from the creation of BiasKG
- mlruns: output from mlflow library, that is used to track experiments. To see the experiments, run `mlflow server`

Other important files:
- graphrag.py: the main file, run this to test the GraphRAG system

## Other notes
- This README file still needs to be updated to include more details
- The requirements.txt file might not be fully up to date (needs to be tested)
