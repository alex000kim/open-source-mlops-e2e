import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import pandas as pd
from alibi_detect.cd import TabularDrift
from alibi_detect.saving import save_detector
from joblib import load
from utils.load_params import load_params

def train_drift_detector(params):
    processed_data_dir = Path(params.data_split.processed_data_dir)
    model_dir = Path(params.train.model_dir)
    model_path = Path(params.train.model_path)
    model = load(model_path)

    X_test = pd.read_pickle(processed_data_dir/'X_test.pkl')
    X_train = pd.read_pickle(processed_data_dir/'X_train.pkl')
    X = pd.concat([X_test, X_train])

    feat_names = X.columns.tolist()
    preprocessor = model[:-1]
    categories_per_feature = {i:None for i,k in enumerate(feat_names) if k.startswith('cat__')}
    cd = TabularDrift(X, 
                    p_val=.05, 
                    preprocess_fn=preprocessor.transform,
                    categories_per_feature=categories_per_feature)

    detector_path = model_dir/'drift_detector'
    save_detector(cd, detector_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = load_params(params_path=args.config)
    train_drift_detector(params)