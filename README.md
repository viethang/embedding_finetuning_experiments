## Embedding finetuning

This project is to experiment with finetuning embeddings.
It mainly follows the guide from
https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/
The generated train dataset and validation data set are included to facilitate subsequence finetunings.

We also experimented finetuning with multilingual-e5-large-instruct.

### Hit rates
 Model| Hit rate
-----|---------
bge-small-en | 0.7786606129398411
multilingual-e5-large-instruct | 0.757094211123723
multilingual-e5-large-instruct (with instruction) | 0.7786606129398411
multilingual-e5-large |  0.7695800227014756
bge-small-en finetuned | 0.8274687854710556
multilingual-e5-large-instruct finetuned | 0.8183881952326901