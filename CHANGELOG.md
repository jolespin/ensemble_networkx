

#### Completed:
* 2023.07.18 - Fixed issue with `SampleSpecificPerturbationNetwork` not being able to handle `X.index` with a `.name` that was not `NoneType`.  Created a hack to allow `pd.MultiIndex` support (converts to strings and warns). Made `include_reference_for_samplespecific=True` the new default which creates a clone of the reference and uses that as the background network.  Added `is_square` to `Symmetric` object.
* 2022.02.09 - Added support for iGraph and non-fully connected networks. Also added UMAP `fuzzy_simplical_set` graph
* 2021.06.24 - Added `get_weights_from_graph` function
* 2021.06.09 - Fixed `condensed_to_dense` ability to handle self interactions
* 2021.04.21 - Fixed `idx_nodes = pd.Index(sorted(set(groups[lambda x: x == group].index) & set(df_dense.index)))` in `connectivity` function to prepare for pandas deprecation.
* 2021.04.12 - Added `community_detection` wrapper for `python-louvain` and `leidenalg`.  Changed `cluster_modularity` function to `cluster_homogeneity` to not be confused with `modularity` metric used for louvain algorithm.
* 2021.03.09 - Large changes took place in this version.  Removed dependency of HiveNetworkX and moved many non-Hive plot functions/classes to EnsembleNetworkX.  Now HiveNetworkX depends on EnsembleNetworkX which will be the more generalizable extension to NetworkX in the Soothsayer ecosystem while maintaining HiveNetworkX's core object on Hive plots. This version has also incorporated a feature engineering class called `CategoricalEngineeredFeature` that is a generalizable replacement to Soothsayer's PhylogenomicFunctionalComponent (which is being deprecated).
* 2020.07.24 - Added `DifferentialEnsembleAssociationNetwork`
* 2020.07.21 - `SampleSpecificPerturbationNetwork` fit method returns self


#### Pending:  
* Rename `Symmetric` object to something more generalizable to similarity and dissimilarity matrices that do not have to be symmetric or completely connected.
* Should `convert_network` actually be `convert_symmetric`? 
* Since iGraph is a dependency, just make code cleaner without the workarounds for not having it as a dependency