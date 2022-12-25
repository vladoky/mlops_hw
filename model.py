import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class Model():
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'gradboost':
            self.model = GradientBoostingRegressor(n_estimators=50)
        else:
            raise ValueError('unknown model type')

        self.feature_names = ['age', 'bmi', 'children', 'sex_female', 'sex_male',
                              'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest',
                              'region_southeast', 'region_southwest']
        self.target_name = 'charges'

        self.fitted = False
        self.train_score = None
        self.test_score = None

    def _prepare_dataset(self, data_path):
        df = pd.read_csv(data_path)
        df = pd.get_dummies(df)
        X = df.drop([self.target_name], axis=1)
        Y = df[self.target_name]
        return X, Y

    def load_model(self, path):
        with open('linear_model.pickle', 'rb') as f:
            self.model = pickle.load(f)
        self.fitted = True

    def save_model(self, path):
        with open('linear_model.pickle', 'wb') as f:
            pickle.dump(self.model, f)

    def test(self, data_path):
        X, Y = self._prepare_dataset(data_path)
        self.test_score = self.model.score(X, Y)
        return self.test_score

    def fit(self, data_path):
        X, Y = self._prepare_dataset(data_path)
        self.model.fit(X, Y)
        self.fitted = True
        self.train_score = self.model.score(X, Y)

    def predict(self, features):
        v = pd.DataFrame({f: [0] for f in self.feature_names})
        v.iloc[0]['age'] = features['age']
        v.iloc[0]['bmi'] = features['bmi']
        v.iloc[0]['children'] = features['children']
        v.iloc[0]['sex_female'] = int(features['sex'] == 'f')
        v.iloc[0]['sex_male'] = int(features['sex'] == 'm')
        v.iloc[0]['smoker_no'] = int(features['smoker'] == 'false')
        v.iloc[0]['smoker_yes'] = int(features['smoker'] == 'true')
        v.iloc[0]['region_northeast'] = int(features['region'] == 'northeast')
        v.iloc[0]['region_northwest'] = int(features['region'] == 'northwest')
        v.iloc[0]['region_southeast'] = int(features['region'] == 'southeast')
        v.iloc[0]['region_southwest'] = int(features['region'] == 'southwest')
        return self.model.predict(v)
