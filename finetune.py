from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from time import perf_counter

start = perf_counter()
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")


finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="finetuned_models/bge-small-en-finetuned",
    val_dataset=val_dataset,
)

finetune_engine.finetune()

end = perf_counter()

print(f"Finetuning bge-small-en completes after {end - start}s")

start = perf_counter()
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

instruction = "Retrieve the document that answer the following question: "
instruct_train_queries = dict((k, instruction + v) for k,v in train_dataset.queries.items())
instructed_train_dataset = train_dataset.copy()
instructed_train_dataset.queries = instruct_train_queries

instruct_val_queries = dict((k, instruction + v) for k,v in val_dataset.queries.items())
instructed_val_dataset = val_dataset.copy()
instructed_val_dataset.queries = instruct_val_queries

finetune_engine = SentenceTransformersFinetuneEngine(
    instructed_train_dataset,
    model_id="intfloat/multilingual-e5-large-instruct",
    model_output_path="finetuned_models/multilingual-e5-large-instruct-finetuned",
    val_dataset=instructed_val_dataset,
    batch_size=5
)

finetune_engine.finetune()

end = perf_counter()

print(f"Finetuning multilingual-e5-large-instruct completes after {end - start}s")