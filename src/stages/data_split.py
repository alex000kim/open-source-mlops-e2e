import os
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.load_params import load_params

def data_split(params):
    raw_data_dir = Path(params.base.raw_data_dir)
    feat_cols = params.base.feat_cols
    countries = params.base.countries
    targ_col = params.base.targ_col
    random_state = params.base.random_state
    test_size = params.data_split.test_size
    processed_data_dir = Path(params.data_split.processed_data_dir)
    processed_data_dir.mkdir(exist_ok=True)
    data_file_paths = [raw_data_dir/f'Churn_Modelling_{country}.csv'  for country in countries]
    df = pd.concat([pd.read_csv(fpath) for fpath in data_file_paths])
    X, y = df[feat_cols], df[targ_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.to_pickle(processed_data_dir/'X_train.pkl')
    X_test.to_pickle(processed_data_dir/'X_test.pkl')
    y_train.to_pickle(processed_data_dir/'y_train.pkl')
    y_test.to_pickle(processed_data_dir/'y_test.pkl')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_split(params)