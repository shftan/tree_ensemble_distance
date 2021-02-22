# Tree Ensemble Distance
An implementation of the gradient boosted tree distance presented in "Tree Space Prototypes: Another Look at Making Tree Ensembles Interpretable".

To compute the GBT-based proximity matrix on sklearn's Digits dataset, run
`python compute_proximity.py`

Note that there are different ways to weigh the trees in a gradient boosted tree ensemble. Check the `proximityCGB` function in `compute_proximity.py` for different ways that are supported by this code. 

This code relies on scikit-learn version 0.24.1. It may not work with other versions.

If you use this code, please cite "Tree Space Prototypes: Another Look at Making Tree Ensembles Interpretable". S Tan, M Soloviev, G Hooker, and MT Wells. ACM-IMS FODS 2020. https://arxiv.org/abs/1611.07115
