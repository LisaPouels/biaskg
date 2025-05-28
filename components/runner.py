from neo4j_graphrag.generation import GraphRAG
import pandas as pd
import mlflow
from components.evaluate_results import evaluate_results
from components.generation import initialize_llm, build_query_prompt
import progressbar

def run_experiment(model, k, df_prompts, retriever, timestamp, dataset, retrieval_query, retriever_name, retriever_type, perturbation):
    llm, override_sample_size = initialize_llm(model)
    sample_size = override_sample_size or len(df_prompts)

    prompts = df_prompts.iloc[:sample_size]
    df_answers = pd.DataFrame(columns=[
        'context', 'question', 'ans0', 'ans1', 'ans2', 'label',
        'RAG_Answer', 'context_condition', 'question_polarity',
        'category', 'target_loc', 'retriever_result'
    ])

    rag = GraphRAG(retriever=retriever, llm=llm)

    with mlflow.start_run(run_name=f"{model}_{perturbation}_k{k}_{retriever_name}_{retriever_type}_{timestamp}_bbq_experiment"):
        mlflow.log_param("model", model)
        mlflow.log_param("retriever", retriever_name)
        mlflow.log_param("embedder model", "nomic-embed-text")
        mlflow.log_param("retrieval query", retrieval_query)
        mlflow.log_param("retriever type", retriever_type)
        mlflow.log_param("sample size", sample_size)
        mlflow.log_param("k", k)
        mlflow.log_input(dataset)

        bar = progressbar.ProgressBar(maxval=len(prompts))
        bar.start()

        for i, row in prompts.iterrows():
            query_text = build_query_prompt(row['context'], row['question'], row['ans0'], row['ans1'], row['ans2'], perturbation)
            response = rag.search(query_text=query_text, retriever_config={"top_k": k, "query_params": {"k": k}}, return_context=True)

            df_answers = pd.concat([df_answers, pd.DataFrame({
                'context': row['context'],
                'question': row['question'],
                'ans0': row['ans0'],
                'ans1': row['ans1'],
                'ans2': row['ans2'],
                'label': row['label'],
                'RAG_Answer': response.answer,
                'context_condition': row['context_condition'],
                'question_polarity': row['question_polarity'],
                'category': row['category'],
                'target_loc': row['target_loc'],
                'retriever_result': [response.retriever_result.items]
            }, index=[0])], ignore_index=True)
            bar.update(i + 1)

        # Evaluate and log
        metrics = evaluate_results(df_answers)
        for name, val in zip(["overall_accuracy", "accuracy_ambiguous", "accuracy_disambiguous", 
                              "accuracy_cost_bias_nonalignment", "bias_disambig", "bias_ambig"], metrics):
            mlflow.log_metric(name, val)
            df_answers[name.replace("accuracy_", "Accuracy_").replace("bias_", "Bias_")] = val

        df_answers['RAG_Answer'] = df_answers['RAG_Answer'].str.replace('\n', ' ')
        df_answers.to_csv(f"Experiments/3_Prompts/{model}_{perturbation}_k{k}_{retriever_name}_{retriever_type}_{timestamp}_bbq_experiment.csv", index=False)
        # df_answers.to_csv(f"Experiments/{model}_{perturbation}_k{k}_{retriever_name}_{retriever_type}_{timestamp}_bbq_experiment.csv", index=False)
