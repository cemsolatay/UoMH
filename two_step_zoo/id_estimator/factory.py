from .estimator import MLEIDEstimator, GeoMLEIDEstimator, TwoNNIDEstimator

def get_id_estimator(cluster_cfg, writer):

    if cluster_cfg["id_estimator"] == "mle":
        id_estimator = MLEIDEstimator(cluster_cfg, writer)
    elif cluster_cfg["id_estimator"] == "geomle":
        id_estimator = GeoMLEIDEstimator(cluster_cfg, writer)
    elif cluster_cfg["id_estimator"] == "twonn":
        id_estimator = TwoNNIDEstimator(cluster_cfg, writer)
    else:
        raise ValueError(f"Unknown ID estimator {cluster_cfg['id_estimator']}")

    return id_estimator
