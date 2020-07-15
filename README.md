
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
ens = enx.EnsembleAssociationNetwork(name="Iris", node_type="leaf measurement", edge_type="rho", observation_type="specimen")
ens.fit(X, n_iter=100)
# ===========================================================
# EnsembleAssociationNetwork(Name:Iris, method: pairwise_rho)
# ===========================================================
#     * Number of nodes (leaf measurement): 4
#     * Number of edges (rho): 6
#     * Observation type: specimen
#     -------------------------------------------------------
#     | Parameters
#     -------------------------------------------------------
#     * n_iter: 100
#     * sample_size: 92
#     * random_state: 0
#     * memory: 11.047 KB
#     -------------------------------------------------------
#     | Data
#     -------------------------------------------------------
#     * Features (n=150, m=4, memory=5.859 KB)
#     * Ensemble (memory=4.812 KB)
#     * Statistics (['mean', 'median', 'var', 'kurtosis', 'skew', 'normaltest|stat', 'normaltest|p_value'], memory=384 B)

# View summary stats
ens.stats_
# Statistics	mean	median	var	kurtosis	skew	normaltest|stat	normaltest|p_value
# Edges							
# (sepal_length, sepal_width)	0.855421	0.854771	0.000066	0.363698	0.151548	1.472627	0.478876
# (sepal_length, petal_length)	-0.801119	-0.799696	0.000445	-0.039800	-0.169622	0.590829	0.744223
# (sepal_length, petal_width)	-0.803683	-0.804366	0.000045	-0.814222	-0.103616	6.422311	0.040310
# (sepal_width, petal_length)	-0.672675	-0.671192	0.000423	0.065722	-0.447298	3.719766	0.155691
# (petal_width, sepal_width)	-0.973008	-0.972823	0.000014	0.386471	-0.204413	1.898091	0.387110
# (petal_width, petal_length)	0.487433	0.484487	0.000692	0.124931	0.517601	4.929038	0.085050

# View ensemble network
ens.ensemble_
# Edges	(sepal_length, sepal_width)	(sepal_length, petal_length)	(sepal_length, petal_width)	(sepal_width, petal_length)	(petal_width, sepal_width)	(petal_width, petal_length)
# n_iter=100						
# 0	0.849206	-0.793988	-0.799695	-0.672147	-0.971270	0.481301
# 1	0.855512	-0.828456	-0.805577	-0.687045	-0.978446	0.511237
# 2	0.868281	-0.778928	-0.804862	-0.654763	-0.968321	0.459651
# 3	0.859762	-0.812899	-0.808884	-0.682954	-0.974394	0.499853
# 4	0.858101	-0.786517	-0.803897	-0.667418	-0.969415	0.473680
# ...	...	...	...	...	...	...
# 95	0.856910	-0.815229	-0.816463	-0.701705	-0.973710	0.519174
# 96	0.853866	-0.796767	-0.799694	-0.669505	-0.972546	0.480086
# 97	0.858599	-0.749676	-0.799779	-0.645614	-0.963164	0.441183
# 98	0.856120	-0.845143	-0.811837	-0.698145	-0.981585	0.532996
# 99	0.860316	-0.782562	-0.797666	-0.651384	-0.970254	0.458598

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