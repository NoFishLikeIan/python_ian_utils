# python_utils
Some function I use all the time.

### embed_data.py
The function takes in a standard train, test split for classification problems and stores 5 type of embedding in dictionary form:
* PCA with 2 components and with 90% variance
* Unsupervised and supervised UMAP embeddings
* T-SNE with given perplexity

When called with:
```
emb = Emedding()
```
The data already is computed (takes some time) and stored. You can call it with a different level of perplexity for the t-SNE.