from sentence_transformers.evaluation import InformationRetrievalEvaluator
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd

from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

def evaluate(
    dataset,
    embed_model,
    top_k=5,
    verbose=False,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(
        nodes, embed_model=embed_model, show_progress=True
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results


def evaluate_st(
    dataset,
    model_id,
    name,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)

val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

# bge
bge = "local:BAAI/bge-small-en"
bge_val_results = evaluate(val_dataset, bge)

df_bge = pd.DataFrame(bge_val_results)

hit_rate_bge = df_bge["is_hit"].mean()
print("Hit rate bge", hit_rate_bge)
evaluate_st(val_dataset, "BAAI/bge-small-en", name="bge")

# finetuned
finetuned = "local:finetuned_models/bge-small-en-finetuned"
val_results_finetuned = evaluate(val_dataset, finetuned)

df_finetuned = pd.DataFrame(val_results_finetuned)

hit_rate_finetuned = df_finetuned["is_hit"].mean()

print("Hit rate finetuned bge", hit_rate_finetuned)

evaluate_st(val_dataset, "bge-small-en-finetuned", name="bge-finetuned")

# e5_instruct
e5_instruct = "local:intfloat/multilingual-e5-large-instruct"
instruction = "Retrieve the document that answer the following question: "
instruct_val_queries = dict((k, instruction + v) for k,v in val_dataset.queries.items())
instructed_val_dataset = val_dataset.copy()
instructed_val_dataset.queries = instruct_val_queries
val_results_e5 = evaluate(instructed_val_dataset, e5_instruct)
df_e5_instruct = pd.DataFrame(val_results_e5)
hit_rate_e5_instruct = df_e5_instruct["is_hit"].mean()
print("Hit rate e5 instruct", hit_rate_e5_instruct)

# e5
e5 = "local:intfloat/multilingual-e5-large"
val_results_e5 = evaluate(val_dataset, e5)
df_e5 = pd.DataFrame(val_results_e5)
hit_rate_e5 = df_e5["is_hit"].mean()
print("Hit rate e5", hit_rate_e5)



# e5_instruct
e5_instruct_finetuned = "local:finetuned_models/multilingual-e5-large-instruct-finetuned"
instruction = "Retrieve the document that answer the following question: "
instruct_val_queries = dict((k, instruction + v) for k,v in val_dataset.queries.items())
instructed_val_dataset = val_dataset.copy()
instructed_val_dataset.queries = instruct_val_queries
val_results_e5_finetuned = evaluate(instructed_val_dataset, e5_instruct_finetuned)
df_e5_instruct_finetuned = pd.DataFrame(val_results_e5_finetuned)
hit_rate_e5_instruct_finetuned = df_e5_instruct_finetuned["is_hit"].mean()
print("Hit rate finetuned e5 instruct", hit_rate_e5_instruct_finetuned)
