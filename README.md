# Embedding trees

This repository contains the code and datasets for the manuscript "Tree-based visualizations of protein sequence embedding space enables improved functional classification of diverse protein superfamilies".



# Installing dependencies

### Downloading this repository

```
# Download this repository
git clone https://github.com/waylandy/chumby

cd chumby
```
### Installing dependencies with `conda`

If you do not have `conda` installed, you can download from the [Anaconda website](https://www.anaconda.com/).

```
conda env create -f environment.yaml
conda activate chumby
```

If you want to exit the environment:

```
conda deactivate
```

If you want to start over, you can also delete the `conda` environment:

```
conda env remove -n chumby
```

**Notes on PyTorch:** This code uses the PyTorch library which can benefit from GPU acceleration. You can monitor your GPU and check the your CUDA version by running `nvidia-smi -l 1`. PyTorch should always work on CPU, however additional steps may be required to enable CUDA, depending on your GPU model. For more information, see the installation guide on the [PyTorch website](https://pytorch.org/). 


# Tutorials

This repository provides tutorials for generating trees-based visualizations of embedding vectors generated for protein language models. Tutorials are included as computational notebook `.ipynb` files which can be viewed directly on GitHub website. These can be run this code on your own computer using JupyterLab, included in the dependencies.


### Beginner tutorial

To try the beginner's tutorial, open `simplified_tutorial.ipynb`.

To provide a brief overview of the tutorial:
1. Read a fasta file of protein sequences.
2. Generate full-size sequence embedding vectors from protein sequences.
    - In this example, we will be specifically using the ESM-1b protein language model.
3. Derive fixed-size sequence embeddings from the full-size sequence embedding vectors.
    - In this example, we will be specifically using the average of all sequence tokens.
4. Calculate an all-vs-all distance matrix from the fixed-size sequence embeddings.
    - In this example, we will be specifically using cosine distance.
5. Apply the neighbor joining algorithm to calculate a tree from the distance matrix.
6. Plot and view the tree using the `ete3` library.


### Advanced tutorial

The advanced tutorials are a series of four notebooks which need to be completed in order:

- `step1-1_gen_embeddings.ipynb`
    - Read a fasta file and generate embeddings using the ESM-1b language model.
    - Compress and save the generated sequence embeddings into an SQLite database.
    - CUDA and multi-threading can be used for generating embeddings.
- `step1-2_learn_manifolds.ipynb`
    - Try various methods of calculate distance matrices from the embeddings. We will be trying a variety of methods for deriving fixed-size embeddings and a variety of distance metrics.
    - Calculate and save UMAP and Neighbor Joining representions of the embedding space. This will be performed on all distance matrices.
    - Calculate statistics for evaluating local accuracy (trustworthiness), global accuracy (Spearman correlation), and biological relevance (silhouette coefficient).
    - Multiprocessing is used to perform calculations in parallel.
- `step1-3_plot_visualizations.ipynb`
    - Draw plots for UMAP and Neighbor Joining embeddings.
    - Multiprocessing is used to draw plots in parallel.
- `step2-1_determine_vibes.ipynb`
	- Train a VAE on a set of fixed-size embeddings, then generate replicate trees from the re-sampled embeddings.
	- Calculate the branch support values for the tree generated the original embeddings using the re-sampled trees.
	- CUDA and multi-threading can be used for training the VAE. Multiprocessing is used to process trees in parallel.

