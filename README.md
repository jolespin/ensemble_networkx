
### Ensemble NetworkX
High-level [Ensemble](https://en.wikipedia.org/wiki/Ensemble_averaging_(machine_learning)) Network implementations in Python.  Built on top of [NetworkX](https://github.com/networkx/networkx) and [Pandas](https://pandas.pydata.org/).  

#### Dependencies:
Compatible for Python 3.

    panda
    numpy
    scipy
    networkx
    matplotlib
    soothsayer_utils
    compositional
    
#### Citations (Debut):
   
   * Nabwera HM+, Espinoza JL+, Worwui A, Betts M, Okoi C, Sesay AK, Bancroft R, Agbla SC, Jarju S, Bradbury RS, Colley M, Jallow AT, Liu J, Houpt ER, Prentice AM, Antonio M, Bernstein RM, Dupont CL+, Kwambana-Adams BA+. *Interactions between fecal gut microbiome, enteric pathogens, and energy regulating hormones among acutely malnourished rural Gambian children*. EBioMedicine. 2021 Oct 22;73:103644. [doi: 10.1016/j.ebiom.2021.103644](https://doi.org/10.1016/j.ebiom.2021.103644). PMID: 34695658.


#### Install:
```
# Stable release (Preferred)
pip install ensemble_networkx

# Current developmental release
pip install git+https://github.com/jolespin/ensemble_networkx
```

#### Source:
* Migrated from [`soothsayer`](https://github.com/jolespin/soothsayer)


#### Supported metrics: 
* [Compositional data](https://en.wikipedia.org/wiki/Compositional_data) (e.g., counts data, [NGS](https://www.illumina.com/science/technology/next-generation-sequencing.html), etc.)
	* [**Do not use** Pearson, Spearman, Kendall-Tau, Biweight Midcorrelation for compositional data](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075).  
	* Compositionally-valid association metrics: 
		* Partial correlation with basis shrinkage (`pcorr_bshrink`) ([Jin et al. 2022](https://arxiv.org/abs/2212.00496), [Erb 2020](https://www.sciencedirect.com/science/article/pii/S2590197420300082))
		* Proportionality (`rho`) ([Lovell et al. 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075), [Erb 2016](https://link.springer.com/article/10.1007/s12064-015-0220-8))
		* Proportionality (`phi`) ([Lovell et al. 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075), [Erb 2016](https://link.springer.com/article/10.1007/s12064-015-0220-8))
		
* [Binary data](https://en.wikipedia.org/wiki/Binary_data) (e.g., detected vs. not-detected)
	* Matthew's Correlation Coefficient (`mcc`) 

* Miscellaneous data:
	* Pearson's correlation (`pearson`)
	* Spearman's correlation (`spearman`)
	* Kendall-Tau (`kendall`)
	* Biweight midcorrelation (`bicor`)

#### Case studies, tutorials and usage:
Documentation will be released upon version 1.0 once API is stabilized.

* [Multimodal sample-specific perturbation networks and undernutrition modeling from Nabwera & Espinoza et al. 2021](https://github.com/jolespin/projects/blob/main/gambia_gut_undernutrition_microbiome/Nabwera-Espinoza_et_al_2021/Notebooks/markdown_version/Nabwera-Espinoza_et_al_2021.md)

```python
import ensemble_networkx as enx
```

#### Simple case of an [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) ensemble network

Here we randomly sample 100 times, calculate the associations for each draw, and calculate summary statistics for the distributons of association values (i.e., edge weights). 

If you choose `sampling_size` float between `0 < x ≤ 1.0` then the number of samples drawn will be `x * n` where `n` is the number of samples.  If an integer is used then it will grab that number of samples for each draw.


```python
import soothsayer_utils as syu
import numpy as np
from scipy import stats

# Load in data
X,y = syu.get_iris_data(["X", "y"])

# Create ensemble network
ens = enx.EnsembleAssociationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
ens.fit(X=X, metric="pearson",  n_iter=100, sampling_size=1.0, with_replacement=True, stats_summary=[np.median, stats.median_abs_deviation], stats_tests=[stats.normaltest], copy_ensemble=True)
print(ens)

======================================================
EnsembleAssociationNetwork(Name:Iris, Metric: pearson)
======================================================
    * Number of nodes (leaf measurement): 4
    * Number of edges (association): 6
    * Observation type: specimen
    --------------------------------------------------
    | Parameters
    --------------------------------------------------
    * n_iter: 100
    * sampling_size: 150
    * random_state: 0
    * with_replacement: True
    * transformation: None
    * memory: 15.238 KB
    --------------------------------------------------
    | Data
    --------------------------------------------------
    * Features (n=150, m=4, memory=9.930 KB)
    * Ensemble (memory=4.812 KB)
    * Statistics (['median', 'median_abs_deviation', 'CI(5%)', 'CI(95%)', 'normaltest|stat', 'normaltest|p_value'], memory=508 B)

```

Let's look at the "ensemble" which includes all of the associations for each of the permutations.

```python
# View ensemble
print(ens.ensemble_.head())
Edges       (sepal_length, sepal_width)  (petal_length, sepal_length)  \
Iterations                                                              
0                             -0.061921                      0.877956   
1                             -0.096871                      0.868054   
2                             -0.274915                      0.881042   
3                             -0.044374                      0.837341   
4                             -0.010991                      0.887051  
```

Now let's look at the summary statistics that were calculated for each of the edges:

```python
# View statistics
print(ens.stats_.head())

Statistics                      median  median_abs_deviation    CI(5%)  \
Edges                                                                    
(sepal_length, sepal_width)  -0.111914              0.048581 -0.227990   
(petal_length, sepal_length)  0.872573              0.011590  0.845797   
(petal_width, sepal_length)   0.814497              0.010882  0.785130   
(petal_length, sepal_width)  -0.432412              0.036563 -0.522004   
(petal_width, sepal_width)   -0.370789              0.034571 -0.457290   

Statistics                     CI(95%)  normaltest|stat  normaltest|p_value  
Edges                                                                        
(sepal_length, sepal_width)  -0.015495         2.649309            0.265895  
(petal_length, sepal_length)  0.893086         3.434098            0.179595  
(petal_width, sepal_length)   0.844199         8.300075            0.015764  
(petal_length, sepal_width)  -0.327368         5.811593            0.054705  
(petal_width, sepal_width)   -0.261532        11.567172            0.003078  

```

#### Simple case of an ensemble network for binary data using [Matthew's Correlation Coefficient (MCC)](https://en.wikipedia.org/wiki/Phi_coefficient)

Pearson correlation isn't designed for binary data (i.e., True/False or 0/1) so you can use `MCC` instead.

```python
# Create ensemble network using MCC for binary data
n,m = 1000, 100
X_binary = pd.DataFrame(
    data=np.random.RandomState(0).choice([0,1], size=(n,m)),
    index=map(lambda i: f"sample_{i}", range(n)),
    columns=map(lambda j:f"feature_{j}", range(m)),
)
ens_binary = enx.EnsembleAssociationNetwork(name="Binary", edge_type="association")
ens_binary.fit(X=X_binary, metric="mcc",  n_iter=100, stats_summary=[np.mean,np.var], copy_ensemble=True)
print(ens_binary)

====================================================
EnsembleAssociationNetwork(Name:Binary, Metric: mcc)
====================================================
    * Number of nodes (None): 100
    * Number of edges (association): 4950
    * Observation type: None
    ------------------------------------------------
    | Parameters
    ------------------------------------------------
    * n_iter: 100
    * sampling_size: 1000
    * random_state: 0
    * with_replacement: True
    * transformation: None
    * memory: 4.969 MB
    ------------------------------------------------
    | Data
    ------------------------------------------------
    * Features (n=1000, m=100, memory=821.352 KB)
    * Ensemble (memory=3.777 MB)
    * Statistics (['mean', 'var', 'CI(5%)', 'CI(95%)', 'normaltest|stat', 'normaltest|p_value'], memory=399.742 KB)
```

#### Simple case of creating sample-specific perturbation networks for compositional data using [Rho Proportionality](https://pubmed.ncbi.nlm.nih.gov/26762323/) and confidence interval of [2.5, 97.5]

*Iris data is NOT compositional but this is for demonstration since they are positive values.*

Sampling size here is in relation to the `reference` class.  


```python
# Create ensemble network
sspn_rho = enx.SampleSpecificPerturbationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
sspn_rho.fit(X=X, y=y, metric="rho", reference="setosa", n_iter=100, confidence_interval=97.5, copy_ensemble=True)

print(sspn_rho)

==============================================================================================
SampleSpecificPerturbationNetwork(Name:Iris, Reference: Reference(setosa[clone]), Metric: rho)
==============================================================================================
    * Number of nodes (leaf measurement): 4
    * Number of edges (association): 6
    * Observation type: specimen
    ------------------------------------------------------------------------------------------
    | Parameters
    ------------------------------------------------------------------------------------------
    * n_iter: 100
    * sampling_size: 50
    * random_state: 0
    * with_replacement: True
    * transformation: None
    * memory: 787.684 KB
    ------------------------------------------------------------------------------------------
    | Data
    ------------------------------------------------------------------------------------------
    * Features (n=200, m=4, memory=9.930 KB)
    ------------------------------------------------------------------------------------------
    | Intermediate
    ------------------------------------------------------------------------------------------
    * Reference Ensemble (memory=220 B)
    * Sample-specific Ensembles (memory=32.227 KB)
    ------------------------------------------------------------------------------------------
    | Terminal
    ------------------------------------------------------------------------------------------
    * Ensemble (memory=703.125 KB)
    * Statistics (['median', 'median_abs_deviation', 'CI(2.5%)', 'CI(97.5%)', 'normaltest|stat', 'normaltest|p_value'], memory=42.188 KB)

    

# View ensemble
print(*repr(sspn_rho.ensemble_).split("\n")[-4:], sep="\n")

Coordinates:
  * Samples     (Samples) <U8 'iris_0' 'iris_1' ... 'iris_148' 'iris_149'
  * Iterations  (Iterations) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
  * Edges       (Edges) object frozenset({'sepal_length', 'sepal_width'}) ......
  

# View statistics
print(*repr(sspn_rho.stats_).split("\n")[-4:], sep="\n")

Coordinates:
  * Samples     (Samples) object 'iris_0' 'iris_1' ... 'iris_148' 'iris_149'
  * Edges       (Edges) object frozenset({'sepal_length', 'sepal_width'}) ......
  * Statistics  (Statistics) <U20 'median' ... 'normaltest|p_value'

# View SSPN for a particular sample
graph = sspn_rho.to_networkx("iris_50")
list(graph.edges(data=True))[0]

'sepal_length',
 'sepal_width',
 {'median': 0.02589165067420246,
  'median_abs_deviation': 0.018200489248854534,
  'CI(2.5%)': -0.01811839255779411,
  'CI(97.5%)': 0.09388843207188755,
  'normaltest|stat': 4.196194170296813,
  'normaltest|p_value': 0.12268967426224149})
```

Now let's output the perturbation matrix which includes all SSPNs across all the samples using the median values of the perturbation distributions for the weight.

```python
X_perturbation = sspn_rho.to_perturbation(weight='median')
X_perturbation.head()
# Edges	(sepal_length, sepal_width)	(sepal_length, petal_length)	(sepal_length, petal_width)	(sepal_width, petal_length)	(sepal_width, petal_width)	(petal_width, petal_length)
# Samples						
# iris_0	0.000757	-0.001263	-0.000486	-0.001056	-0.000663	0.001352
# iris_1	-0.007569	0.001436	-0.000439	-0.003043	0.001497	-0.000325
# iris_2	0.000189	-0.001334	-0.000158	-0.000911	-0.000124	0.000483
# iris_3	0.000011	-0.005670	0.000814	-0.005226	0.001058	-0.000984
# iris_4	-0.000990	-0.000612	0.000142	-0.002192	-0.001285	0.001501
```

#### Sample-specific perturbation networks for compositional data using [partial correlation with basis shrinkage](https://arxiv.org/abs/2212.00496)

You can also now use `partial_correlation_with_basis_shrinkage` from the [`compositional`](https://github.com/jolespin/compositional) package ([Jin et al. 2022](https://arxiv.org/abs/2212.00496) and [Erb 2020](https://www.sciencedirect.com/science/article/pii/S2590197420300082)). 


```python
sspn_bshrink = enx.SampleSpecificPerturbationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
sspn_bshrink.fit(X=X, y=y, metric="pcorr_bshrink", reference="setosa", n_iter=100, confidence_interval=97.5, copy_ensemble=True)
print(sspn_bshrink)

========================================================================================================
SampleSpecificPerturbationNetwork(Name:Iris, Reference: Reference(setosa[clone]), Metric: pcorr_bshrink)
========================================================================================================
    * Number of nodes (leaf measurement): 4
    * Number of edges (association): 6
    * Observation type: specimen
    ----------------------------------------------------------------------------------------------------
    | Parameters
    ----------------------------------------------------------------------------------------------------
    * n_iter: 100
    * sampling_size: 50
    * random_state: 0
    * with_replacement: True
    * transformation: None
    * memory: 787.684 KB
    ----------------------------------------------------------------------------------------------------
    | Data
    ----------------------------------------------------------------------------------------------------
    * Features (n=200, m=4, memory=9.930 KB)
    ----------------------------------------------------------------------------------------------------
    | Intermediate
    ----------------------------------------------------------------------------------------------------
    * Reference Ensemble (memory=220 B)
    * Sample-specific Ensembles (memory=32.227 KB)
    ----------------------------------------------------------------------------------------------------
    | Terminal
    ----------------------------------------------------------------------------------------------------
    * Ensemble (memory=703.125 KB)
    * Statistics (['median', 'median_abs_deviation', 'CI(2.5%)', 'CI(97.5%)', 'normaltest|stat', 'normaltest|p_value'], memory=42.188 KB)
```

#### Create a SSPN using a custom association function

Here we specify a custom function for the associations which is the inverse kullback leibler divergence.

```python
# Custom association function
def inverse_kullbackleibler(a,b, base=2):
    return 1/stats.entropy(pk=a, qk=b, base=base)

# Create ensemble network
sspn_kl = enx.SampleSpecificPerturbationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
sspn_kl.fit(X=X, y=y, metric=inverse_kullbackleibler, reference="setosa", n_iter=100, function_is_pairwise=False, copy_ensemble=True)

print(sspn_kl)

==================================================================================================================
SampleSpecificPerturbationNetwork(Name:Iris, Reference: Reference(setosa[clone]), Metric: inverse_kullbackleibler)
==================================================================================================================
    * Number of nodes (leaf measurement): 4
    * Number of edges (association): 6
    * Observation type: specimen
    --------------------------------------------------------------------------------------------------------------
    | Parameters
    --------------------------------------------------------------------------------------------------------------
    * n_iter: 100
    * sampling_size: 50
    * random_state: 0
    * with_replacement: True
    * transformation: None
    * memory: 787.684 KB
    --------------------------------------------------------------------------------------------------------------
    | Data
    --------------------------------------------------------------------------------------------------------------
    * Features (n=200, m=4, memory=9.930 KB)
    --------------------------------------------------------------------------------------------------------------
    | Intermediate
    --------------------------------------------------------------------------------------------------------------
    * Reference Ensemble (memory=220 B)
    * Sample-specific Ensembles (memory=32.227 KB)
    --------------------------------------------------------------------------------------------------------------
    | Terminal
    --------------------------------------------------------------------------------------------------------------
    * Ensemble (memory=703.125 KB)
    * Statistics (['median', 'median_abs_deviation', 'CI(5%)', 'CI(95%)', 'normaltest|stat', 'normaltest|p_value'], memory=42.188 KB)
```

#### Feature engineering using categories

Let's engineer some categories by collapsing by some predefined category. Check out `Phylogenomic Functional Categories` in [Espinoza et al. 2022](https://academic.oup.com/pnasnexus/article/1/5/pgac239/6762943) for how these are used in practice.


```python
from soothsayer_utils import get_iris_data
from scipy import stats
import pandas as pd
import ensemble_networkx as enx

X, y = get_iris_data(["X","y"])
# Usage
cef = enx.CategoricalEngineeredFeature(name="Iris", observation_type="sample")

# Add categories
category_1 = pd.Series(X.columns.map(lambda x:x.split("_")[0]), X.columns)
cef.add_category(
    name_category="leaf_type", 
    mapping=category_1,
)
# Optionally add scaling factors, statistical tests, and summary statistics
# Compile all of the data
cef.compile(scaling_factors=X.sum(axis=0), stats_tests=[stats.normaltest])
# Unpacking engineered groups: 100%|██████████| 1/1 [00:00<00:00, 2974.68it/s]
# Organizing feature sets: 100%|██████████| 4/4 [00:00<00:00, 17403.75it/s]
# Compiling synopsis [Basic Feature Info]: 100%|██████████| 2/2 [00:00<00:00, 32768.00it/s]
# Compiling synopsis [Scaling Factor Info]: 100%|██████████| 2/2 [00:00<00:00, 238.84it/s]

# View the engineered features
cef.synopsis_
#   initial_features        number_of_features      leaf_type(level:0)      scaling_factors sum(scaling_factors)    mean(scaling_factors)   sem(scaling_factors)    std(scaling_factors)
# leaf_type                                                         
# sepal     [sepal_width, sepal_length]     2       sepal   [458.6, 876.5]  1335.1  667.55  208.95  208.95
# petal     [petal_length, petal_width]     2       petal   [563.7, 179.90000000000003]     743.6   371.8   191.9   191.9

# Transform a dataset using the defined categories
cef.fit_transform(X, aggregate_fn=np.sum)
# leaf_type sepal   petal
# sample            
# iris_0    8.6     1.6
# iris_1    7.9     1.6
# iris_2    7.9     1.5
# iris_3    7.7     1.7
```

#### Cluster networks using Leiden or Louvain community detection
We are going to run Leiden community detection but since it is stochastic and not deterministic, we are going to use 100 different random seeds and only consider clusters that consistent (i.e., `minimum_cooccurrence_rate=1.0`)

```python
# Get graph
graph = enx.convert_network(X.T.corr(), nx.Graph)

# Cluster graph
cn = enx.ClusteredNetwork(name="Iris", node_type="leaf measurement", edge_type="pearson")

# Fit model
cn.fit(graph, n_iter=100, algorithm="leiden", minimum_cooccurrence_rate=1.0)
print(cn)

===========================================
ClusteredNetwork(Name:Iris, weight_dtype: float64)
===========================================
    * Algorithm: leiden
    * Number of iterations: 100
    * Minimum edge cooccurrence rate: 1.0
    * Number of nodes clustered (leaf measurement): 150 (100.00%)
    * Number of edges clustered (pearson): 6126 (54.82%)
    ---------------------------------------
    | Cluster Sizes (N = 2)
    ---------------------------------------
    Leiden_1    99
    Leiden_2    51
```

Let's take a look at the cluster assignments: 

```python
cn.node_to_cluster_.head()

Nodes[Clustered]
iris_1    Leiden_2
iris_0    Leiden_2
iris_2    Leiden_2
iris_3    Leiden_2
iris_4    Leiden_2
Name: Clusters[Expanded], dtype: object
```

We can also get a cluster to nodes dictionary:

```python
cn.cluster_to_nodes_

Leiden_1    {iris_117, iris_76, iris_59, iris_126,...
Leiden_2    {iris_0, iris_12, iris_43, iris_4, ...
Name: Clusters[Collapsed], dtype: object
```

We can also get the clustered graph in other formats: 

```python
# Default
graph = cn.graph_clustered_

weights = cn.to_pandas_series()

df_adjacency = cn.to_pandas_dataframe(vertical=False)

df_edgelist = cn.to_pandas_dataframe(vertical=True)

sym = cn.to_symmetric()

```

#### Differential ensemble association networks
We are going to create a differential between setosa and not-setosa samples.

```python
X,y = syu.get_iris_data(["X", "y"])
y_setosaornot = y.map(lambda x: {True:"setosa", False:"not_setosa"}[x == "setosa"])

# Differential network between setosa and not setosa
dn = enx.DifferentialEnsembleAssociationNetwork(name="Iris")
dn.fit(X, y_setosaornot, reference="setosa", treatment="not_setosa", metric="rho")
print(dn)

========================================================================================================
DifferentialEnsembleAssociationNetwork(Name:Iris, Reference: setosa, Treatment: not_setosa, Metric: rho)
========================================================================================================
    * Number of nodes (None): 4
    * Number of edges (None): 6
    * Observation type: None
    ----------------------------------------------------------------------------------------------------
    | Parameters
    ----------------------------------------------------------------------------------------------------
    * n_iter: 1000
    * sampling_size: 50
    * random_state: 0
    * with_replacement: True
    * transformation: None
    * memory: 105.539 KB
    ----------------------------------------------------------------------------------------------------
    | Data
    ----------------------------------------------------------------------------------------------------
    * Features (n=150, m=4, memory=9.930 KB)
    ----------------------------------------------------------------------------------------------------
    | Intermediate
    ----------------------------------------------------------------------------------------------------
    * Reference Ensemble (memory=47.496 KB)
    * Treatment Ensemble (memory=47.496 KB)
    ----------------------------------------------------------------------------------------------------
    | Terminal
    ----------------------------------------------------------------------------------------------------
    * Initial Statistics (['median', 'median_abs_deviation', 'CI(5%)', 'CI(95%)', 'normaltest|stat', 'normaltest|p_value'])
    * Comparative Statistics (['wasserstein_distance', 'mannwhitneyu|stat', 'mannwhitneyu|p_value'], memory=364 B)
    * Differential Statistics (['median'], memory=268 B)

```

We can look at individual stats for each class: 

```python
dn.ensemble_reference_.stats_

Statistics	median	median_abs_deviation	CI(5%)	CI(95%)	normaltest|stat	normaltest|p_value
Edges						
(sepal_width, sepal_length)	0.786607	0.043967	0.655605	0.866409	125.652750	5.186232e-28
(petal_length, sepal_length)	0.460913	0.090514	0.236955	0.662517	10.686368	4.780625e-03
(petal_width, sepal_length)	-0.600863	0.019191	-0.644792	-0.548529	33.238974	6.056873e-08
(sepal_width, petal_length)	0.261425	0.095604	0.012081	0.483333	11.792565	2.749648e-03
(sepal_width, petal_width)	-0.599378	0.031315	-0.673050	-0.514876	25.564313	2.810476e-06
(petal_length, petal_width)	-0.513017	0.033226	-0.592368	-0.425776	7.823623	2.000423e-02

dn.ensemble_treatment_.stats_

Statistics	median	median_abs_deviation	CI(5%)	CI(95%)	normaltest|stat	normaltest|p_value
Edges						
(sepal_width, sepal_length)	0.341311	0.051657	0.217615	0.460566	1.117749	0.571852
(petal_length, sepal_length)	-0.019897	0.054220	-0.151335	0.109942	0.923136	0.630295
(petal_width, sepal_length)	-0.781452	0.017277	-0.820192	-0.729637	25.201274	0.000003
(sepal_width, petal_length)	-0.596316	0.037117	-0.678457	-0.493624	23.866671	0.000007
(sepal_width, petal_width)	-0.627446	0.027775	-0.687832	-0.555307	11.590521	0.003042
(petal_length, petal_width)	0.046047	0.043560	-0.063424	0.154860	0.228881	0.891865

```

We can also look stats that compare the distributions of each edge between the 2 conditions: 

```python
dn.stats_comparative_

	wasserstein_distance	mannwhitneyu|stat	mannwhitneyu|p_value
Edges			
(sepal_width, sepal_length)	0.434245	999756.0	0.000000e+00
(petal_length, sepal_length)	0.477957	998548.0	0.000000e+00
(petal_width, sepal_length)	0.179624	999994.0	0.000000e+00
(sepal_width, petal_length)	0.851155	1000000.0	0.000000e+00
(sepal_width, petal_width)	0.028351	675085.0	7.047032e-42
(petal_length, petal_width)	0.556989	0.0	0.000000e+00
```

Lastly, we can get the differentials between 2 statistics calculated for each conditions.  We would use this as our resulting differential network: 


```python
dn.stats_differential_

	median
Edges	
(sepal_width, sepal_length)	-0.445296
(petal_length, sepal_length)	-0.480810
(petal_width, sepal_length)	-0.180589
(sepal_width, petal_length)	-0.857740
(sepal_width, petal_width)	-0.028068
(petal_length, petal_width)	0.559064
```

