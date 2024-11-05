import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
from optuna import create_study
from optuna.integration.mlflow import MLflowCallback
from optuna.trial import FrozenTrial
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score


def evaluate_thresholds(
    thresholds: npt.NDArray[np.float32],
    y_true: npt.NDArray[np.float32],
    y_pred_proba: npt.NDArray[np.float32],
    plot: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """evaluates thresholds and find optimal precision recall and f1 score and plot them

    Args:
        thresholds (npt.NDArray[np.float32]): _description_
        y_true (npt.NDArray[np.float32]): _description_
        y_pred_proba (npt.NDArray[np.float32]): _description_
        plot (bool, optional): _description_. Defaults to True.

    Returns:
        tuple[list[float], list[float], list[float]]: _description_
    """
    rcs = []
    prs = []
    f1s = []

    for t in thresholds:
        test_binary_pred = y_pred_proba[:, 1] >= t
        prs.append(precision_score(y_true, test_binary_pred))
        rcs.append(recall_score(y_true, test_binary_pred))
        f1s.append(f1_score(y_true, test_binary_pred))

    metrics_df = pd.DataFrame({"threshold": thresholds, "score": f1s, "metric": "F1"})
    metrics_df = pd.concat(
        (
            metrics_df,
            pd.DataFrame({"threshold": thresholds, "score": rcs, "metric": "Recall"}),
        )
    )
    metrics_df = pd.concat(
        (
            metrics_df,
            pd.DataFrame(
                {"threshold": thresholds, "score": prs, "metric": "Precision"}
            ),
        )
    )

    optimal_thr = thresholds[np.argmax(f1s)]
    optimal_f1 = f1s[np.argmax(f1s)]
    optimal_rc = rcs[np.argmax(f1s)]
    optimal_pr = prs[np.argmax(f1s)]

    print("Threshold with Max F1 Score: ", optimal_thr)
    print(f"F1 at threshold {optimal_thr}: {optimal_f1}")
    print(f"Recall at threshold {optimal_thr}: {optimal_rc}")
    print(f"Precision at threshold {optimal_thr}: {optimal_pr} ")

    if plot:
        fig = px.line(
            metrics_df,
            x="threshold",
            y="score",
            color="metric",
            title="Metrics per Threshold",
        )
        fig.show()

    return rcs, prs, f1s
