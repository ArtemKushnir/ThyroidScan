import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

scalers = {"standard": StandardScaler, "min_max": MinMaxScaler, "max_abs": MaxAbsScaler, "robust": RobustScaler}


def scale_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_name: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    scaler = scalers[scaler_name]()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)
