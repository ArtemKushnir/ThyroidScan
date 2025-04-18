from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


class TrigonometricFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, is_add_sum: bool = False, is_add_product: bool = False, keep_original: bool = True) -> None:
        self.is_add_sum = is_add_sum
        self.is_add_product = is_add_product
        self.keep_original = keep_original

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: Optional[pd.DataFrame | np.ndarray] = None
    ) -> "TrigonometricFeatures":
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        input_is_dataframe = isinstance(X, pd.DataFrame)

        if input_is_dataframe:
            X_df = cast(pd.DataFrame, X)
            column_names = X_df.columns
            X_values = X_df.values
        else:
            X_values = X

        n_cols = X_values.shape[1]
        features_per_col = 2 + self.is_add_sum + self.is_add_product
        if self.keep_original:
            features_per_col += 1
        n_new_cols = n_cols * features_per_col

        X_new = np.zeros((X_values.shape[0], n_new_cols))

        col_idx = 0
        for i in range(n_cols):
            col_values = X_values[:, i]

            if self.keep_original:
                X_new[:, col_idx] = col_values
                col_idx += 1

            X_new[:, col_idx] = np.sin(col_values)
            col_idx += 1
            X_new[:, col_idx] = np.cos(col_values)
            col_idx += 1

            if self.is_add_sum:
                X_new[:, col_idx] = np.sin(col_values) + np.cos(col_values)
                col_idx += 1

            if self.is_add_product:
                X_new[:, col_idx] = np.sin(col_values) * np.cos(col_values)
                col_idx += 1

        if input_is_dataframe:
            X_df = cast(pd.DataFrame, X)
            new_columns = []
            for col in column_names:
                if self.keep_original:
                    new_columns.append(col)
                new_columns.append(f"{col}_sin")
                new_columns.append(f"{col}_cos")
                if self.is_add_sum:
                    new_columns.append(f"{col}_sin_cos_sum")
                if self.is_add_product:
                    new_columns.append(f"{col}_sin_cos_product")
            return pd.DataFrame(X_new, columns=new_columns, index=X_df.index)

        return X_new

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray, y: Optional[pd.DataFrame | np.ndarray] = None, **fit_params: Any
    ) -> pd.DataFrame | np.ndarray:
        self.fit(X)
        return self.transform(X)


class FeaturesTransform:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, columns: list[str] = None) -> None:
        if X_train.columns != X_test.columns:
            raise ValueError("the columns in X_train and X_test are different")
        self.columns = columns if columns is not None else X_train.columns
        self.X_train = X_train
        self.X_test = X_test

    def perform_poly(self, degree: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train_selected = self.X_train[self.columns] if self.columns is not None else self.X_train
        X_test_selected = self.X_test[self.columns] if self.columns is not None else self.X_test

        poly_transform = PolynomialFeatures(degree=degree)
        X_train_poly = poly_transform.fit_transform(X_train_selected)
        X_test_poly = poly_transform.transform(X_test_selected)

        feature_names = poly_transform.get_feature_names_out(X_train_selected.columns)

        train_poly_df = pd.DataFrame(X_train_poly, columns=feature_names, index=self.X_train.index)
        test_poly_df = pd.DataFrame(X_test_poly, columns=feature_names, index=self.X_test.index)

        columns_to_drop = []

        for col in self.columns:
            if col in feature_names:
                columns_to_drop.append(col)

        train_poly_df = train_poly_df.drop(columns=columns_to_drop, errors="ignore")
        test_poly_df = test_poly_df.drop(columns=columns_to_drop, errors="ignore")

        X_train_result = pd.concat([self.X_train, train_poly_df], axis=1)
        X_test_result = pd.concat([self.X_test, test_poly_df], axis=1)

        return X_train_result, X_test_result

    def perform_pca(self, n_components: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        pca = PCA() if n_components is None else PCA(n_components)
        pca.fit(self.X_train)
        return pca.transform(self.X_train), pca.transform(self.X_test)

    def preform_trigonometry(
        self, is_add_sum: bool = False, is_add_product: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train_selected = self.X_train[self.columns] if self.columns is not None else self.X_train
        X_test_selected = self.X_test[self.columns] if self.columns is not None else self.X_test

        trig_transform = TrigonometricFeatures(is_add_sum, is_add_product, False)

        X_train_trig = trig_transform.fit_transform(X_train_selected)
        X_test_trig = trig_transform.transform(X_test_selected)

        X_train_result = pd.concat([self.X_train, X_train_trig], axis=1)
        X_test_result = pd.concat([self.X_test, X_test_trig], axis=1)

        return X_train_result, X_test_result
