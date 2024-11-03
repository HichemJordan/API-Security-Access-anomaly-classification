import numpy as np
import plotly.express as px
import polars as pl
import ppscore as pps


def aggregate_node_features(
    data: pl.DataFrame, node_features: list[str], by: str = "_id"
) -> pl.DataFrame:
    """Utility function to generate basic aggregation statistics features for node level features

    Args:
        data (pl.DataFrame): input dataframe
        node_features (list[str]): list of node features to aggregate
        by (str, optional): the graph ID column. Defaults to "_id".

    Returns:
        pl.DataFrame: dataframe with aggregated features
    """
    aggs = []
    for f in node_features:
        avg = pl.col(f).mean().alias(f"avg_{f}")
        min_val = pl.col(f).min().alias(f"min_{f}")
        max_val = pl.col(f).max().alias(f"max_{f}")
        std = pl.col(f).std().alias(f"std_{f}")
        aggs += [avg, min_val, max_val, std]
    agg_data = data.group_by(by).agg(aggs)

    return agg_data
