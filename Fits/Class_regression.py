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

class PowerModelWithFixedPoints:
    def __init__(self, b_fixed, fixed_points):
        """
        b_fixed : Exposant fixe (puissance)
        fixed_points : Liste de tuples (x, y) représentant les points fixes
        """
        self.b_fixed = b_fixed
        self.fixed_points = fixed_points  # Liste de points fixes [(x1, y1), (x2, y2), ...]
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Ajuste le modèle en tenant compte des points fixes.
        """
        # Transformation des caractéristiques : x^b_fixed
        X_trans = np.power(X, self.b_fixed).reshape(-1, 1)

        # Ajout d'une colonne pour le biais (intercept)
        X_design = np.hstack([X_trans, np.ones((X_trans.shape[0], 1))])

        # Ajout des contraintes des points fixes
        fixed_X = np.array([pt[0] for pt in self.fixed_points])
        fixed_y = np.array([pt[1] for pt in self.fixed_points])
        fixed_X_trans = np.power(fixed_X, self.b_fixed).reshape(-1, 1)
        fixed_X_design = np.hstack([fixed_X_trans, np.ones((fixed_X_trans.shape[0], 1))])

        # Concaténation des données d'entraînement et des points fixes
        X_combined = np.vstack([X_design, fixed_X_design])
        y_combined = np.concatenate([y, fixed_y])

        # Poids : 1 pour les données normales, poids très élevé pour les points fixes
        weights = np.concatenate([np.ones(len(X)), 1e6 * np.ones(len(fixed_X))])
        
        # Régression pondérée : ajustement avec les poids
        W = np.diag(weights)
        beta = np.linalg.inv(X_combined.T @ W @ X_combined) @ (X_combined.T @ W @ y_combined)
        
        # Stockage du coefficient et calcul explicite de l'intercept
        self.coef_ = beta[0]
        self.intercept_ = beta[1]

    def predict(self, X):
        """
        Prédit les valeurs cibles pour les nouvelles données.
        """
        X_trans = np.power(X, self.b_fixed)
        return self.coef_ * X_trans + self.intercept_

    def score(self, X, y):
        """
        Calcule le score R² ou une métrique personnalisée (par exemple RMSE).
        """
        y_pred = self.predict(X)
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
