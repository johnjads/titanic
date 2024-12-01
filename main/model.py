import numpy as np
import pandas as pd
import qwak
# Important to call run_local when using the Build SDK
from qwak.model.tools import run_local
from catboost import CatBoostClassifier, Pool, cv
from catboost.datasets import titanic
from qwak.model.base import QwakModel
from sklearn.model_selection import train_test_split


class TitanicSurvivalPrediction(QwakModel):
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=1000,
            custom_loss=["Accuracy"],
            loss_function="Logloss",
            learning_rate=None,
        )

    def build(self):
        titanic_train, _ = titanic()
        titanic_train.fillna(-999, inplace=True)

        x = titanic_train.drop(["Survived", "PassengerId"], axis=1)
        y = titanic_train.Survived

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.85, random_state=42
        )

        # mark categorical features
        cate_features_index = np.where(x_train.dtypes != float)[0]

        self.model.fit(
            x_train,
            y_train,
            cat_features=cate_features_index,
            eval_set=(x_test, y_test),
        )

        # Cross validating the model (5-fold)
        cv_data = cv(
            Pool(x, y, cat_features=cate_features_index),
            self.model.get_params(),
            fold_count=5,
        )

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["PassengerId"], axis=1)
        return pd.DataFrame(
            self.model.predict_proba(df[self.model.feature_names_])[:, 1],
            columns=['Survived_Probability']
        )