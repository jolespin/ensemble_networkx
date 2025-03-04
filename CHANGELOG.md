
#### Completed:

* 2025.3.4 - Added `pairwise_log_ratio` function to compute pairwise log ratios between components
* 2025.3.3 - Added `read_parquet_nonredundant_pairwise_matrix` and `write_parquet_nonredundant_pairwise_matrix`
* 2025.2.11 - Optimized `community_membership_cooccurrence` to use boolean and no broadcasting
* 2025.2.11 - Added `get_undirected_igraph_edgelist_indices` and `get_undirected_igraph_connected_components`
* 2025.2.11 - Moved code to `__init__.py` and added `parallel_backend="threading"` to `community_detection`
* 2025.2.11 - Change `edge_cluster_cooccurrence` to `community_membership_cooccurrence` because they are node pairs not edges
* 2025.2.10 - Optimized `edge_cluster_cooccurrence`
* 2025.2.10 - `community_detection` now allows for `igraph` inputs and is threaded
* 2025.2.10 - If `data` and `into` are same type in `convert_network` then input is returned
* 2024.7.xx - Changed `n_iter` default from 100 to 1000 in `ClusteredNetwork`.
* 2024.7.xx - Added `BiDirectionalClusteredNetwork` which is separates positive and negative weighted edges, clusters the graphs separately, then merges the clustered representations.
* 2024.7.xx - Calculate `hubs` for `ClusteredNetwork`
* 2024.7.xx - Added `node_connectivity_clustered_` and `cluster_connectivity_` to `ClusteredNetwork`
* 2024.7.xx - Added `nodes_ordering` to `Symmetric` and uses this in `convert_network`.  To avoid situations where the order changes between conversions of `pd.DataFrame` and `Symmetric` objects, these conversions must be done explicitly. 
* 2024.5.30 - Added `grouped_node_connectivity_from_numpy`, `grouped_node_connectivity_from_pandas_dataframe`, `group_connectivity_from_numpy`, and `group_connectivity_from_pandas_dataframe`.  Also rebuilt `connectivity` so it uses these functions and added support for masking outliers. Now able to output either node or group-level connectivities.
* 2024.2.5 - Fixed `__repr__` for unfitted `ClusteredNetwork` objects.
* 2024.2.5 - Fixed error from indexing using set objects (`X_subset = X[initial_features]`)
* 2023.9.25 - Added `AggregateNetwork` class, `evenness/entropy` calculations, and `.mad` for median absolute devation to `Symmetry`.
* 2023.9.5 - Changed `method="biweight_midcorrelation"` to `method="bicor"`.  Changed default method to `pearson` instead of `rho` to generalize (though, please use `rho`, `phi`, or `pcorr_bshrink` for compositional data).  Added `partial_correlation_with_basis_shrinkage` support from `comositional` package using `method="pcorr_bshrink"` to use similar terminology with `Propr` and `ppcorr` R packages.
* 2023.8.15 - Added `ClusteredNetwork` for wrapper around `community_detection` and `edge_cluster_cooccurrence` (formerly known as `cluster_homogeneity`).
* 2023.8.14 - Changed `dense` to `redundant` to be more consistent with `scikit-bio`.  Added `confidence_interval` to ensemble networks.  Changed default metrics to `np.median` and `stats.median_abs_deviation`.  Changed default `sampling_size` from `0.618...` to `1.0` and `with_replacement=False` to `with_replacement=True`.
* 2023.7.20 - Added `pairwise_mcc` with Mathew's Correlation Coefficient for binary correlations. Functionality also available in `EnsembleAssociationNetwork` ([@411an13](https://github.com/411an13))
* 2023.7.18 - Fixed issue with `SampleSpecificPerturbationNetwork` not being able to handle `X.index` with a `.name` that was not `NoneType`.  Created a hack to allow `pd.MultiIndex` support (converts to strings and warns). Made `include_reference_for_samplespecific=True` the new default which creates a clone of the reference and uses that as the background network.  Added `is_square` to `Symmetric` object.
* 2022.2.9 - Added support for iGraph and non-fully connected networks. Also added UMAP `fuzzy_simplical_set` graph
* 2021.6.24 - Added `get_weights_from_graph` function
* 2021.6.9 - Fixed `condensed_to_dense` ability to handle self interactions
* 2021.4.21 - Fixed `idx_nodes = pd.Index(sorted(set(groups[lambda x: x == group].index) & set(df_dense.index)))` in `connectivity` function to prepare for pandas deprecation.
* 2021.4.12 - Added `community_detection` wrapper for `python-louvain` and `leidenalg`.  Changed `cluster_modularity` function to `cluster_homogeneity` to not be confused with `modularity` metric used for louvain algorithm.
* 2021.3.9 - Large changes took place in this version.  Removed dependency of HiveNetworkX and moved many non-Hive plot functions/classes to EnsembleNetworkX.  Now HiveNetworkX depends on EnsembleNetworkX which will be the more generalizable extension to NetworkX in the Soothsayer ecosystem while maintaining HiveNetworkX's core object on Hive plots. This version has also incorporated a feature engineering class called `CategoricalEngineeredFeature` that is a generalizable replacement to Soothsayer's PhylogenomicFunctionalComponent (which is being deprecated).
* 2020.7.24 - Added `DifferentialEnsembleAssociationNetwork`
* 2020.7.21 - `SampleSpecificPerturbationNetwork` fit method returns self


#### Pending:  
* Add option to include confidence intervals and MAD to graph with `convert_network`
* Add `weight` attribute to `convert_network`
* Move arguments in `.fit` to `__init__` to better reflect usage in `scikit-learn`.
* Since iGraph is a dependency, just make code cleaner without the workarounds for not having it as a dependency
* Use `edge_weights_` and `node_weights_` in `Symmetric` objects like with `AggregateNetworks`? Are `Symmetric` objects immutable? Don't want node connectivity to be calclulated, the underlying network modified, and then will be inaccurate.