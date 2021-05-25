# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from sklearn.metrics import classification_report
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: dict) -> list:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["example_test_data_ratio"]
    )
    # y_train = y_train.to_numpy()
    # y_test = y_test.to_numpy()
    return [X_train,X_test,y_train,y_test]


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LogisticRegression(max_iter = 10000)
    regressor.fit(X_train, y_train.values.ravel())
    return regressor


## General model output nodes

def generate_classification_report(regressor: LogisticRegression, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Generate classification report for model
    :param model: model object
    :param X_test: Pandas DataFrame
    :param y_test: Pandas Dataframe of test target values
    :return: Classification Report
    """
    y_pred = regressor.predict(X_test)
    logger = logging.getLogger(__name__)
    logger.info(classification_report(y_test, y_pred))

# def evaluate_model(
#     regressor: LogisticRegression, X_test: pd.DataFrame, y_test: pd.DataFrame):
#     """Calculates and logs the coefficient of determination.

#     Args:
#         regressor: Trained model.
#         X_test: Testing data of independent features.
#         y_test: Testing data for price.
#     """
#     y_pred = regressor.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     logger = logging.getLogger(__name__)
#     logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
