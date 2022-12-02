# Embedding trees

This repository contains the code and datasets for the manuscript "Tree-based visualizations of protein sequence embedding space enables improved functional classification of diverse protein superfamilies".


## Instructions

We will be running code within a conda environment. If you do not have Anaconda installed, download it at (https://www.anaconda.com)

Run the following commands in your BASH terminal.

```
# Create a conda environment using environment.yml
# The environment will be created under the name "embedding_trees", defined in the yml file

conda env create -f environment.yml

# Activate the new conda environment

conda activate chumby

# Enter the directory and start Jupyter Lab (included in the environment).

cd embedding_trees
jupyter lab

# Inside the directory, run the notebooks in order.
# More specific instructions are provided in each notebook.
```

If you want to start over, you can delete the conda environment with the following BASH command.

```
conda env remove -n chumby
```


**Important note for newer GPUs:**

Newer GPUs may require a later version of PyTorch in order to enable CUDA acceleration.
We recommend upgrading to an appropriate version of PyTorch using `pip` inside the `conda` environment.
For more information on installing PyTorch, visit (https://pytorch.org).

If you choose not to do this, the code should still be able to run in CPU (albeit slower).



