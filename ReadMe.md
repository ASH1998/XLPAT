# Clustering

Approaches : 
1. Semi Supervised KeyWord Extraction + Brown Clustering
2. Semi Supervised KeyWord Extraction + Distil-Bert Embeddings + HDBSCAN


### Reason of two Approaches : 
1. Clustering using K-means and heirarchical clustering didn't yeild good results. The elbow method and silhouette clustering did not help in finding a perfect number of clusters.

![elbow](/Static/elbow.jpg)

2. HDBSCAN was able to find a perfect cluster as shown in below image, but thinking about practicality of this solution, each problem can be atributted with multiple clusters, so Brown Clustering was applied to this problem.

![hdbscan](/Static/hdbscan.jpg)


### Structure of Files
- Each directory has its own model and output results.
- `model.py` - the main file containing the model and implementation
- `(for brown clustering)` An utils file containing Brown Clustering implementation has been used. This code is partly taken from https://github.com/yangyuan/brown-clustering/
- pickle files of objects of models are stored in respective folders for further use.

### Dependencies:
(full dependency has been attached into requirements.txt)
- python 3.8
- nltk
- tqdm
- pandas
- pickle
- matplotlib
- yake : `pip install git+https://github.com/LIAAD/yake`
- sentence-transformers : `pip install sentence-transformers`
- hdbscan : `pip install hdbscan`
- umap : `pip install umap-learn`

### Usage
 `python model.py`

 To check similar words/ similar clustered words for Brown Clustering : 
 `clustering.get_similar('vibration')`

 ![example](/Static/brown1.jpg)


### Output
1. Brown Clustering
`clustered_Dataset.csv`
The column clusters contains the index of `cluster.csv`.
This indicates each text can be given an attribute of multiple clusters.

2. HDBSCAN + DistilBert Embeddings
`hdbscan_Dataset.csv `
The clusters column has been categorised into 11 clusters, which was derived from unsupervised number of clusters from using UMAP dimensionality reduction and HDBSCAN.

### Possible Improvements: 
- Testing with new samples with Bert and Xlnet Embeddings.
- Evaluating the sanity of clusters, this was not possible with current sample size.






