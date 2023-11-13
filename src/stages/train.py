import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.load_params import load_params
from xgboost import XGBClassifier

def train(params):
    processed_data_dir = Path(params.data_split.processed_data_dir)
    random_state = params.base.random_state
    feat_cols = params.base.feat_cols
    model_type = params.train.model_type
    model_dir = Path(params.train.model_dir)
    model_dir.mkdir(exist_ok=True)
    model_path = Path(params.train.model_path)
    train_params = params.train.params

    X_train = pd.read_pickle(processed_data_dir/'X_train.pkl')
    y_train = pd.read_pickle(processed_data_dir/'y_train.pkl')

    if model_type == "randomforest":
        clf = RandomForestClassifier(random_state=random_state,
                                **train_params)
    elif model_type == "xgboost":
        clf = XGBClassifier(random_state=random_state,
                                    **train_params)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feat_cols)]
    )
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("clf", clf)]
        )

    model.fit(X_train, y_train)
    dump(model, model_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = load_params(params_path=args.config)
    train(params)