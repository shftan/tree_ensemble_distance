#!/usr/bin/python3
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import augmented_gb 


def proximityCGB(
    model,
    X,
    train_indexes,
    use_gammas=True,
    use_gammas_squared=False,
    use_prediction_scores=True,
    use_variances=True):
    # This will work for up to a few tens of thousands of samples,
    # after that it will blow up and we will ahve to have more memory efficient
    # (but slower) approaches.
    N = len(X)
    proximity = np.zeros((N, N), dtype=np.float32)
    if type(X) is np.ndarray:
        X_train = X[train_indexes]
    else:
        X_train = X.iloc[train_indexes]
    norm = 0
    for tree_idx, tree_ in enumerate(tqdm(model.estimators_,
                                          dynamic_ncols=True,
                                          total=len(model.estimators_),
                                          desc='Computing proximity matrix')):
        # Each estimator is actually an array of trees, for multiclass settings
        # We are interested only in the first one for the time being.
        tree = tree_[0]
        # Same computation on each tree
        # First, get the terminal leaves of each sample
        terminal_leaves = tree.apply(X)
        # Get which samples share the same terminal leaf.
        # This computation, the way it is done, requires to store an
        #  NxN matrix even if the output is sparse. Can be done manually for
        # less memory but more compute.
        proximity_mask = (terminal_leaves[:, None] ==
                            terminal_leaves[None, :]).astype(np.float32)

        # We now need to weight that mask. We consider 3 sources of weighting.
        # - Gammas (which will be one if gammas are disabled at training)
        # - Prediction scores of the tree
        # - Other sources of weighting, for example the estimator number
        # (not implemened yet)
        #
        # The weight of a pair is the product of the weights of the samples,
        # so we can do the same trick as before, compute the weights per
        # sample, compute the outer product, and multiply by the binary mask.
        weights = np.ones(N, dtype=np.float32)
        gammas = np.array([model.all_gammas[tree_idx][tl]
                           for tl in terminal_leaves], dtype=np.float32)
        if use_gammas:
            weights *= gammas
        if use_gammas_squared:
            if use_gammas:
                weights *= gammas
            else:
                weights *= (gammas**2)

        if use_prediction_scores:
            # Multiply by the predictions
            weights *= np.abs(tree.predict(X))

        if use_variances:
            weights *= np.var(tree.predict(X_train))

        # Any other weighting goes here
        # XXXX

        # Normalize by the max of the weights.
        # In the v(g^2) case the weights are constant per tree anyway.
        # For the predictions they change per sample, so take the max to normalize.
        norm += np.abs(weights).max()
        # Get the outer product and multiply by the mask. Then add to the output
        # Again, this can be more efficient.
        proximity += proximity_mask * weights[:, None]
    return proximity / norm


# Parse input
parser = argparse.ArgumentParser(description='Build random forest proximity for a digits data set.')
parser.add_argument(
    '--gamma_type', 
    type=str, 
    choices=['disabled', 'per_tree', 'per_leaf'],
    default="per_tree",
    help='Compute gamma per leaf, per tree, or do not compute gamma at all'
)
args = parser.parse_args()
args.gamma_type = augmented_gb.GammaType[args.gamma_type]

# Load digits data (only 4 and 9) and split in train and test
X, y = load_digits(return_X_y=True)
keep = np.where(np.logical_or(y == 4, y==9))[0]
X = X[keep]
y = y[keep]
indexes = range(len(y))
X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(X, y, indexes, test_size=0.4, random_state=12345)
indexes_train = np.sort(indexes_train)
indexes_test = np.sort(indexes_test)

# Train GBT. This is a modified GBT that inherits from sklearn's
# GradientBoostingClassifier and that tracks the gammas used at each tree/leaf.
# We can set up if the gammas operate at the tree level or at the leaf level
# (or if they are not used at all)
model = augmented_gb.AugmentedGradientBoostingClassifier(
	n_estimators=50,
	learning_rate=0.1,
	max_depth=3,
	random_state=0,
	gamma_type=args.gamma_type
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Model ROC AUC score: {100 * roc_auc_score(y_test, y_pred):.2f}")

# Compute and print the proximity matrix
prox = proximityCGB(
	model=model,
	X=X,
	train_indexes=indexes_train,
    use_gammas=True,
    use_gammas_squared=False,
    use_prediction_scores=False,
    use_variances=True
)
print("Proximity matrix:")
print(np.round(prox, 3))