from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

scalers = {
    "standard": StandardScaler,
    "min_max": MinMaxScaler,
    "max_abs": MaxAbsScaler,
    "robust": RobustScaler
}

def scale_data(X_train, X_test, scaler_name: str = "standard"):
    scaler = scalers[scaler_name]()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)