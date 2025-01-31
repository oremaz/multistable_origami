# <Creates different classes for polynomial fits of a function>
# Copyright (C) 2025 Orel Mazor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class ConstantModel:
    def __init__(self):
        self.coef = None
        self.intercept = None
        self.model = (
            self.constant_function
        )  # Attribuer la fonction constante à l'attribut model

    def constant_function(self, X):
        return np.full((X.shape[0],), self.intercept)

    def fit(self, X, y):
        self.coef = 0
        self.intercept = np.mean(y)  # La constante prédite est la moyenne de y

    def predict(self, X):
        # Utiliser la fonction constante pour faire des prédictions
        return self.model(X)

    def score(self, X, y):
        # Calculer le RMSE pour un modèle constant
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))  # Calculer le RMSE
        return -rmse  # Retourner la valeur négative du RMSE


class PowerModel:
    def __init__(self, b_fixed):
        self.b_fixed = b_fixed
        self.model = LinearRegression()
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        # Transformation des caractéristiques : x^b_fixed
        X_poly = np.power(X, self.b_fixed)

        # S'assurer que X_poly a la forme correcte pour LinearRegression
        if len(X_poly.shape) == 1:
            X_poly = X_poly.reshape(-1, 1)

        # Ajuster le modèle de régression linéaire
        self.model.fit(X_poly, y)

        # Extraire les coefficients
        self.coef = self.model.coef_[0]  # Coefficient de x^b
        self.intercept = self.model.intercept_  # Intercept (c)

    def predict(self, X):
        # Transformation des caractéristiques pour la prédiction
        X_poly = np.power(X, self.b_fixed)
        if len(X_poly.shape) == 1:
            X_poly = X_poly.reshape(-1, 1)
        return self.model.predict(X_poly)

    def score(self, X, y):
        # Calculer le R²
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))  # Calculer le RMSE
        return -rmse  # Retourner la valeur négative du RMSE


class PowerModelWithFixedPoint:
    def __init__(self, b_fixed, fixed_point_x, fixed_point_y):
        self.b_fixed = b_fixed
        self.fixed_point_x = fixed_point_x
        self.fixed_point_y = fixed_point_y
        self.model = LinearRegression(
            fit_intercept=False
        )  # Pas d'intercept automatique
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        # Transform features: (x^b_fixed - fixed_point_x^b_fixed)
        X_poly = np.power(X, self.b_fixed) - np.power(self.fixed_point_x, self.b_fixed)

        # Fit the linear regression model sans intercept
        self.model.fit(X_poly.reshape(-1, 1), y)

        # Store the coefficient and explicitly set intercept to force fixed point
        self.coef = self.model.coef_[0]
        # Store the coefficient and explicitly set intercept to force fixed point
        self.intercept = (
            self.fixed_point_y
            - self.coef * np.power(self.fixed_point_x, self.b_fixed)
            + np.power(self.fixed_point_x, self.b_fixed)
        )

    def predict(self, X):
        # Transform features for prediction: (x^b_fixed - fixed_point_x^b_fixed)
        X_poly = np.power(X, self.b_fixed) - np.power(self.fixed_point_x, self.b_fixed)

        # Predict without adjusting the intercept
        return self.model.predict(X_poly.reshape(-1, 1)) + self.intercept

    def score(self, X, y):
        # Calculate R² using predictions adjusted for the fixed point
        y_pred = self.predict(X)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        return -rmse


class PolynomialModel:
    def __init__(self, degree):
        self.degree = degree
        self.model = LinearRegression()
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        # Transformation des caractéristiques en caractéristiques polynomiales
        X_poly = self.poly_features.fit_transform(X)  # Transformer X

        # Ajuster le modèle de régression linéaire
        self.model.fit(X_poly, y)

        # Extraire les coefficients et l'ordonnée à l'origine
        self.coef = self.model.coef_  # Coefficients du polynôme
        self.intercept = self.model.intercept_  # Intercept (c)

    def predict(self, X):
        # Transformation des caractéristiques pour la prédiction
        X_poly = self.poly_features.transform(X)  # Transformer X
        return self.model.predict(X_poly)  # Prédictions

    def score(self, X, y):
        # Calculer le RMSE
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))  # Calculer le RMSE
        return -rmse  # Retourner la valeur négative du RMSE
