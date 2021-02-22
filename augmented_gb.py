#!/usr/bin/python3

# This file extends sklearn's BinomialDeviance and GradientBoostingClassifier
# classes to keep track of the scaling gammas in gradient boosting trees.
# Additionally, the default GradientBoostingClassifier uses independent gammas
# per leaf. The AugmentedGradientBoostingClassifier also allows one to have 
# scaling gammas at the tree level, or to not use scaling gammas at all,
# two common variations of gradient boosted trees not directly supported
# by sklearn

from sklearn.ensemble import _gb_losses
from sklearn.ensemble import _gb

from sklearn.base import is_classifier
from sklearn.tree._tree import TREE_LEAF
from sklearn.utils.validation import _deprecate_positional_args

from collections import OrderedDict
import numpy as np

from enum import Enum  
GammaType = Enum('GammaType', 'disabled per_leaf per_tree')


""" An augmented BinomialDeviance loss that can apply gammas at the leaf level
or tree level (or to disable the gamma scaling completely)"""
class AugmentedBinomialDeviance(_gb_losses.BinomialDeviance):
    
    def __init__(self, n_classes, gamma_type=GammaType.per_leaf):
        self.gamma_type = gamma_type
        super().__init__(n_classes=n_classes)
    
    # Override update_terminal_regions to be able to apply gammas at the leaf level
    # or tree level (or to disable the gamma scaling completely)
    def update_terminal_regions(self, tree, X, y, residual, raw_predictions,
                                sample_weight, sample_mask,
                                learning_rate=0.1, k=0):
        """Update the terminal regions (=leaves) of the given tree and
        updates the current predictions of the model. Traverses tree
        and invokes template method `_update_terminal_region`.

        Parameters
        ----------
        tree : tree.Tree
            The tree object.
        X : ndarray of shape (n_samples, n_features)
            The data array.
        y : ndarray of shape (n_samples,)
            The target labels.
        residual : ndarray of shape (n_samples,)
            The residuals (usually the negative gradient).
        raw_predictions : ndarray of shape (n_samples, K)
            The raw predictions (i.e. values from the tree leaves) of the
            tree ensemble at iteration ``i - 1``.
        sample_weight : ndarray of shape (n_samples,)
            The weight of each sample.
        sample_mask : ndarray of shape (n_samples,)
            The sample mask to be used.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree by
             ``learning_rate``.
        k : int, default=0
            The index of the estimator being updated.

        """
        # compute leaf for each sample in ``X``.
        terminal_regions = tree.apply(X)

        # mask all which are not in sample mask.
        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        # update each leaf (= perform line search)
        if self.gamma_type == GammaType.per_tree:
            numerator = len(residual)
            denominator = np.sum((y - residual) * (1 - y + residual))
            global_gamma = numerator / denominator
            for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
                tree.value[leaf, :, :] *= global_gamma
        elif self.gamma_type == GammaType.per_leaf:
            for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
                self._update_terminal_region(tree, masked_terminal_regions,
                                            leaf, X, y, residual,
                                            raw_predictions[:, k], sample_weight)
        elif self.gamma_type == GammaType.disabled:
            pass

        # update predictions (both in-bag and out-of-bag)
        raw_predictions[:, k] += \
            learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis=0)

"""This is a modified GBT that inherits from sklearn's
GradientBoostingClassifier and that tracks the gammas used at each tree/leaf.
We can set up if the gammas operate at the tree level or at the leaf level
(or if they are not used at all)"""
class AugmentedGradientBoostingClassifier(_gb.GradientBoostingClassifier):
    @_deprecate_positional_args
    def __init__(self, *, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 validation_fraction=0.1, n_iter_no_change=None, tol=1e-4,
                 ccp_alpha=0.0, 
                 gamma_type=GammaType.per_leaf):

        assert loss == 'deviance'
        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
        self.gamma_type = gamma_type
        self.all_gammas = []
        

    # Override _check params so we can assign our custom loss that is gamma aware.
    def _check_params(self):
        super()._check_params()
        assert is_classifier(self)
        self.loss_ = AugmentedBinomialDeviance(self.n_classes_, gamma_type=self.gamma_type)
    
    
    # Override _fit_stage so we can store the gammas
    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
                   random_state, X_csc=None, X_csr=None):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self.loss_
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)

            # induce regression tree on residuals
            tree = _gb.DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha)

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight,
                     check_input=False)

            pred_pre = tree.predict(X)
            # The chosen terminal leaf does not change after the
            # gamma scaling, so we can compute this here
            winning_leaf_per_sample = tree.apply(X).copy()

            # update tree leaves
            loss.update_terminal_regions(
                tree.tree_, X, y, residual, raw_predictions, sample_weight,
                sample_mask, learning_rate=self.learning_rate, k=k)

            # Save the gammas
            pred_post = tree.predict(X)
            gamma_per_sample = pred_post/pred_pre
            # Given a terminal node, the gammas of all the samples
            # assigned to that node should be the same.
            terminal_leaves = np.unique(winning_leaf_per_sample)
            self.all_gammas.append(OrderedDict())
            for leaf in terminal_leaves:
                gammas = gamma_per_sample[np.where(terminal_leaves == leaf)[0]]
                assert np.all(np.abs((gammas - gammas[:, None])) < 1e-5), "Not all gammas are equal!"
                self.all_gammas[i][leaf] = gammas[0]

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions