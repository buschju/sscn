from typing import Tuple

import mlflow
import numpy
import pandas as pd
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from torch.nn import Module
from torch.utils.data import Dataset


def threshold_coefficient_matrix(coefficient_matrix: numpy.ndarray,
                                 threshold: float,
                                 ) -> numpy.ndarray:
    """
    Applies columns-wise thresholding to a given coefficient matrix. For each columns, only the largest absolute values
    are kept as long as their cumulated sum is <= ro * column_sum. Smaller entries are set to zero.
    Source: https://github.com/panji1990/Deep-subspace-clustering-networks
    """
    if threshold < 1:
        N = coefficient_matrix.shape[1]
        Cp = numpy.zeros((N, N))
        S = numpy.abs(numpy.sort(-numpy.abs(coefficient_matrix), axis=0))
        Ind = numpy.argsort(-numpy.abs(coefficient_matrix), axis=0)
        for i in range(N):
            cL1 = numpy.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while not stop:
                csum = csum + S[t, i]
                if csum > threshold * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = coefficient_matrix[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = coefficient_matrix

    return Cp


def post_process_coefficient_matrix(coefficient_matrix: numpy.ndarray,
                                    num_clusters: int,
                                    max_cluster_dim: int,
                                    noise_threshold: float,
                                    noise_alpha: float,
                                    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Applies post-processing to a given coefficient matrix. After thresholding, the matrix is post-processed and cluster
    labels are extracted using Spectral Clustering.
    Source: https://github.com/panji1990/Deep-subspace-clustering-networks
    """
    # Threshold
    coefficient_matrix = threshold_coefficient_matrix(coefficient_matrix=coefficient_matrix,
                                                      threshold=noise_threshold
                                                      )

    # Post-processing and spectral clustering
    n = coefficient_matrix.shape[0]
    coefficient_matrix = 0.5 * (coefficient_matrix + coefficient_matrix.T)
    coefficient_matrix = coefficient_matrix - numpy.diag(numpy.diag(coefficient_matrix)) + numpy.eye(n, n)
    r = max_cluster_dim * num_clusters + 1
    U, S, _ = svds(coefficient_matrix, r, v0=numpy.ones(n))
    U = U[:, ::-1]
    S = numpy.sqrt(S[::-1])
    S = numpy.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = numpy.abs(Z ** noise_alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = SpectralClustering(n_clusters=num_clusters,
                                  eigen_solver='arpack',
                                  affinity='precomputed',
                                  assign_labels='discretize',
                                  n_jobs=-1,
                                  )
    grp = spectral.fit_predict(L) + 1

    return grp, L


def match_labels(y_true: numpy.ndarray,
                 y_pred: numpy.ndarray,
                 ) -> numpy.ndarray:
    """
    Source: https://github.com/panji1990/Deep-subspace-clustering-networks
    """
    Label1 = numpy.unique(y_true)
    nClass1 = len(Label1)
    Label2 = numpy.unique(y_pred)
    nClass2 = len(Label2)
    nClass = numpy.maximum(nClass1, nClass2)
    G = numpy.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pred == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = numpy.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = numpy.array(index)
    c = index[:, 1]
    newL2 = numpy.zeros(y_pred.shape)
    for i in range(nClass2):
        newL2[y_pred == Label2[i]] = Label1[c[i]]
    return newL2


def get_clustering_error(y_true: numpy.ndarray,
                         y_pred: numpy.ndarray,
                         ) -> float:
    """
    Source: https://github.com/panji1990/Deep-subspace-clustering-networks
    """
    c_x = match_labels(y_true, y_pred)
    err_x = numpy.sum(y_true[:] != c_x[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    return missrate


def evaluate_model(model: Module,
                   dataset: Dataset,
                   **get_cluster_assignments_kwargs,
                   ) -> Tuple[float, float, float]:
    y_true = numpy.array([data[1] for data in dataset])
    y_pred = model.get_cluster_assignments(dataset=dataset,
                                           **get_cluster_assignments_kwargs,
                                           )

    acc = 1. - get_clustering_error(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')

    return acc, ari, nmi


def export_from_mlflow(mlflow_uri: str,
                       mlflow_experiment_name: str,
                       metrics: Tuple[str, ...],
                       ) -> pd.DataFrame:
    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    # Get experiment by ID
    experiment = client.get_experiment_by_name(name=mlflow_experiment_name)
    experiment_id = experiment.experiment_id

    # Load parameters and metrics
    results_df = []
    for run in client.search_runs(experiment_ids=[experiment_id]):
        run_id = run.info.run_id

        data = run.data.params
        data.update({key: run.data.metrics[key] for key in run.data.metrics.keys() if key in metrics})
        run_df = pd.DataFrame(data=data,
                              index=[run_id],
                              )

        results_df += [run_df]

    results_df = pd.concat(results_df,
                           sort=True,
                           )

    return results_df
