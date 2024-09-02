import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from eli5.sklearn import PermutationImportance
from joblib import load
from sklearn.metrics import (confusion_matrix, f1_score, make_scorer,
                            roc_auc_score)
from utils.load_params import load_params

def eval(params):
    processed_data_dir = Path(params.data_split.processed_data_dir)
    model_path = Path(params.train.model_path)
    feat_cols = params.base.feat_cols
    random_state = params.base.random_state

    X_test = pd.read_pickle(processed_data_dir/'X_test.pkl')
    y_test = pd.read_pickle(processed_data_dir/'y_test.pkl')
    model = load(model_path)
    y_prob = model.predict_proba(X_test).astype(float)
    y_prob = y_prob[:, 1]
    y_pred = y_prob >= 0.5

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    metrics = {
        'f1': f1,
        'roc_auc': roc_auc
    }
    json.dump(
        obj=metrics,
        fp=open('metrics.json', 'w'),
        indent=4,
        sort_keys=True
    )

    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    fig_dir = reports_dir/'figures'
    fig_dir.mkdir(exist_ok=True)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
    plt.savefig(fig_dir/'cm.png')

    out_feat_names = model[:-1].get_feature_names_out(feat_cols)
    preprocessor = model.named_steps['preprocessor']
    clf = model.named_steps['clf']
    X_test_transformed = preprocessor.transform(X_test)

    perm = PermutationImportance(clf, scoring=make_scorer(f1_score), random_state=random_state).fit(X_test_transformed, y_test)

    feat_imp = zip(out_feat_names, perm.feature_importances_)
    df_feat_imp = pd.DataFrame(feat_imp, columns=[
        'feature',
        'importance'])
    df_feat_imp = df_feat_imp.sort_values(by='importance', ascending=False)
    feat_importance_fpath = reports_dir/'feat_imp.csv'
    df_feat_imp.to_csv(feat_importance_fpath, index=False, float_format='%.2f')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = load_params(params_path=args.config)
    eval(params)