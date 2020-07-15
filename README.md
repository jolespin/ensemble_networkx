
### Ensemble NetworkX
High-level [Ensemble](https://en.wikipedia.org/wiki/Ensemble_averaging_(machine_learning)) Network implementations in Python.  Built on top of [NetworkX](https://github.com/networkx/networkx) and [Pandas](https://pandas.pydata.org/).  

#### Dependencies:
Compatible for Python 3.

    pandas >= 1
    numpy
    scipy >= 1
    networkx >= 2
    matplotlib >= 3
    hive_networkx >= 2020.06.30
    soothsayer_utils >= 2020.07.01
    compositional >= 2020.05.19

#### Install:
```
# "Stable" release (still developmental)
pip install ensemble_networkx
# Current release
pip install git+https://github.com/jolespin/ensemble_networkx
```

#### Source:
* Migrated from [`soothsayer`](https://github.com/jolespin/soothsayer)

#### Usage:

```python
import ensemble_networkx as enx
```

#### Simple case of an [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) ensemble network

```python
# Load in data
import soothsayer_utils as syu
X = syu.get_iris_data(["X"])

# Create ensemble network
ens = enx.EnsembleAssociationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
ens.fit(X=X, metric="spearman",  n_iter=100, stats_summary=[np.mean,np.var, stats.kurtosis, stats.skew], stats_tests=[stats.normaltest], copy_ensemble=True)
print(ens)
# =======================================================
# EnsembleAssociationNetwork(Name:Iris, Metric: spearman)
# =======================================================
#     * Number of nodes (leaf measurement): 4
#     * Number of edges (association): 6
#     * Observation type: specimen
#     ---------------------------------------------------
#     | Parameters
#     ---------------------------------------------------
#     * n_iter: 100
#     * sampling_size: 92
#     * random_state: 0
#     * with_replacement: False
#     * transformation: None
#     * memory: 16.156 KB
#     ---------------------------------------------------
#     | Data
#     ---------------------------------------------------
#     * Features (n=150, m=4, memory=10.859 KB)
#     * Ensemble (memory=4.812 KB)
#     * Statistics (['mean', 'var', 'kurtosis', 'skew', 'normaltest|stat', 'normaltest|p_value'], memory=496 B)

# View ensemble
print(ens.ensemble_.head())
# Edges       (sepal_width, sepal_length)  (sepal_length, petal_length)  \
# Iterations                                                              
# 0                             -0.113835                      0.880407   
# 1                             -0.243982                      0.883397   
# 2                             -0.108511                      0.868627   
# 3                             -0.151437                      0.879405   
# 4                             -0.241807                      0.869027  

# View statistics
print(ens.stats_.head())
# Statistics                        mean       var  kurtosis      skew  \
# Edges                                                                  
# (sepal_width, sepal_length)  -0.167746  0.002831  0.191176  0.287166   
# (sepal_length, petal_length)  0.880692  0.000268 -0.107437  0.235619   
# (petal_width, sepal_length)   0.834140  0.000442 -0.275487 -0.219778   
# (sepal_width, petal_length)  -0.304403  0.003472 -0.363377  0.059179   
# (sepal_width, petal_width)   -0.285237  0.003466 -0.606118  0.264103 

```

#### Simple case of creating sample-specific perturbation networks


```python
# Create ensemble network
sspn_rho = enx.SampleSpecificPerturbationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
sspn_rho.fit(X=X, y=y, metric="rho", reference="setosa", n_iter=100, stats_summary=[np.mean,np.var], copy_ensemble=True)

print(sspn_rho)
# ============================================================================
# SampleSpecificPerturbationNetwork(Name:Iris, Reference: setosa, Metric: rho)
# ============================================================================
#     * Number of nodes (leaf measurement): 4
#     * Number of edges (association): 6
#     * Observation type: specimen
#     ------------------------------------------------------------------------
#     | Parameters
#     ------------------------------------------------------------------------
#     * n_iter: 100
#     * sampling_size: 30
#     * random_state: 0
#     * with_replacement: False
#     * transformation: None
#     * memory: 518.875 KB
#     ------------------------------------------------------------------------
#     | Data
#     ------------------------------------------------------------------------
#     * Features (n=150, m=4, memory=10.859 KB)
#     ------------------------------------------------------------------------
#     | Intermediate
#     ------------------------------------------------------------------------
#     * Reference Ensemble (memory=208 B)
#     * Sample-specific Ensembles (memory=20.312 KB)
#     ------------------------------------------------------------------------
#     | Terminal
#     ------------------------------------------------------------------------
#     * Ensemble (memory=468.750 KB)
#     * Statistics (['mean', 'var', 'normaltest|stat', 'normaltest|p_value'], memory=18.750 KB)
# Coordinates:
#   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
#   * Iterations  (Iterations) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
#   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
# Coordinates:
#   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
#   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
#   * Statistics  (Statistics) <U18 'mean' 'var' ... 'normaltest|p_value'

# View ensemble
print(*repr(sspn_rho.ensemble_).split("\n")[-4:], sep="\n")
# Coordinates:
#   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
#   * Iterations  (Iterations) int64 0 1 2 3 4 5 6 7 8 ... 92 93 94 95 96 97 98 99
#   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})

# View statistics
print(*repr(sspn_rho.stats_).split("\n")[-4:], sep="\n")
# Coordinates:
#   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
#   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
#   * Statistics  (Statistics) <U18 'mean' 'var' ... 'normaltest|p_value'

# View SSPN for a particular sample
graph = sspn_rho.to_networkx("iris_50")
list(graph.edges(data=True))[0]
# ('sepal_width',
#  'sepal_length',
#  {'mean': 0.04004398613232575,
#   'var': 0.0011399047127054046,
#   'normaltest|stat': 6.063925790957182,
#   'normaltest|p_value': 0.04822089259665142})

```

#### Create a SSPN using a custom association function

```python
# Custom association function
def inverse_kullbackleibler(a,b, base=2):
    return 1/stats.entropy(pk=a, qk=b, base=base)

# Create ensemble network
sspn_kl = enx.SampleSpecificPerturbationNetwork(name="Iris", node_type="leaf measurement", edge_type="association", observation_type="specimen")
sspn_kl.fit(X=X, y=y, metric=inverse_kullbackleibler, reference="setosa", n_iter=100, stats_summary=[np.mean,np.var], function_is_pairwise=False, copy_ensemble=True)

print(sspn_kl)
# ================================================================================================
# SampleSpecificPerturbationNetwork(Name:Iris, Reference: setosa, Metric: inverse_kullbackleibler)
# ================================================================================================
#     * Number of nodes (leaf measurement): 4
#     * Number of edges (association): 6
#     * Observation type: specimen
#     --------------------------------------------------------------------------------------------
#     | Parameters
#     --------------------------------------------------------------------------------------------
#     * n_iter: 100
#     * sampling_size: 30
#     * random_state: 0
#     * with_replacement: False
#     * transformation: None
#     * memory: 518.875 KB
#     --------------------------------------------------------------------------------------------
#     | Data
#     --------------------------------------------------------------------------------------------
#     * Features (n=150, m=4, memory=10.859 KB)
#     --------------------------------------------------------------------------------------------
#     | Intermediate
#     --------------------------------------------------------------------------------------------
#     * Reference Ensemble (memory=208 B)
#     * Sample-specific Ensembles (memory=20.312 KB)
#     --------------------------------------------------------------------------------------------
#     | Terminal
#     --------------------------------------------------------------------------------------------
#     * Ensemble (memory=468.750 KB)
#     * Statistics (['mean', 'var', 'normaltest|stat', 'normaltest|p_value'], memory=18.750 KB)

# View ensemble
print(*repr(sspn_kl.ensemble_).split("\n")[-4:], sep="\n")
# Coordinates:
#   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
#   * Iterations  (Iterations) int64 0 1 2 3 4 5 6 7 8 9
#   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})

# View statistics
print(*repr(sspn_kl.stats_).split("\n")[-4:], sep="\n")
# Coordinates:
#   * Samples     (Samples) object 'iris_50' 'iris_51' ... 'iris_148' 'iris_149'
#   * Edges       (Edges) object frozenset({'sepal_width', 'sepal_length'}) ... frozenset({'petal_width', 'petal_length'})
#   * Statistics  (Statistics) <U18 'mean' 'var' ... 'normaltest|p_value'

# View SSPN for a particular sample
graph = sspn_kl.to_networkx("iris_50")
list(graph.edges(data=True))[0]
# ('sepal_width',
#  'sepal_length',
#  {'mean': -130.40664983969182,
#   'var': 2083.5807070609085,
#   'normaltest|stat': 15.2616635290025,
#   'normaltest|p_value': 0.00048525707083011354})
```