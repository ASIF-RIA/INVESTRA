import numpy as np


def weighted_ensemble(pred_xgb, pred_lstm, pred_prophet, weights=(0.5, 0.3, 0.2)):
    w1, w2, w3 = weights
    n = min(len(pred_xgb), len(pred_lstm), len(pred_prophet))
    return (
        w1 * np.asarray(pred_xgb)[-n:]
        + w2 * np.asarray(pred_lstm)[-n:]
        + w3 * np.asarray(pred_prophet)[-n:]
    )
