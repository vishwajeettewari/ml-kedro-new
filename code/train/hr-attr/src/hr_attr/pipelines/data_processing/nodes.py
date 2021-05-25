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

from typing import Any, Dict
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kedro.extras.datasets.matplotlib import MatplotlibWriter

def make_plot(hr_data: pd.DataFrame):
    
    fig,ax=plt.subplots()
    hr_data.groupby('Education').Age.count().plot.bar()
    fig.set_size_inches(12,12)
    return fig

def dummies(hr_data: pd.DataFrame)-> list:
    p=pd.get_dummies(hr_data['EnvironmentSatisfaction'] , prefix='ES')
    hr_data=pd.concat([hr_data,p],axis=1)
    hr_data.drop('EnvironmentSatisfaction',axis=1,inplace=True)
    logger = logging.getLogger(__name__)
    logger.info('EnvironmentSatisfaction dummied up')
    return [hr_data,p]

def plot_dummies(data: pd.DataFrame):
    fig,ax=plt.subplots()
    sns.countplot(data['ES_4'])
    fig.set_size_inches(12,12)
    return fig
