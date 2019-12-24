import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import os
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']
#merge_f=pd.read_csv('merge_fre.csv')
train_all=pd.concat([train[features],test[features]],axis=0)
train_all=train_all.reset_index(drop=True)
train_f_all=pd.DataFrame()
for f in features:
    train_f_all['me_'+f]=train_all[f]*merge_f['fre_'+f]
train_f_all.shape
le = LabelEncoder()
ff=[c for c in train_f_all.columns if 'me_' in c]
for f in ff:
    train_f_all[f]=le.fit_transform(train_f_all[f])
features = [c for c in train.columns if c not in ['ID_code', 'target']]
train_new=train[features]
train_new=pd.concat([train_new,train_f_all[:200000]],axis=1)
train_new.shape
train_new
features = [c for c in train.columns if c not in ['ID_code', 'target']]
test_new=test[features]
test_f=train_f_all[200000:].reset_index(drop=True)
test_new=pd.concat([test_new,test_f],axis=1)

test_new.shape


test_new

param = {
    'bagging_freq': 10, #handling overfitting
    'bagging_fraction': 0.2,#handling overfitting - adding some noise
     #'boost': 'dart', 
    #'boost': 'goss',
     'boost_from_average':False,
     'boost': 'gbdt',   
    'feature_fraction': 0.15, #handling overfitting
    'learning_rate': 0.01, #the changes between one auc and a better one gets really small thus a small learning rate performs better
    'max_depth':2, 
    'metric':'auc',
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'xentropy', 
    'verbosity':1,
    "bagging_seed" : 122,
    "seed": 20,
    }

num_folds = 11

folds = StratifiedKFold(n_splits=num_folds, random_state=627)
oof = np.zeros(len(train))
getVal = np.zeros(len(train))
predictions = np.zeros(len(target))


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_new.values, target.values)):
    
    X_train, y_train = train_new.iloc[trn_idx], target.iloc[trn_idx]
    X_valid, y_valid = train_new.iloc[val_idx], target.iloc[val_idx]
    
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(train_new.iloc[val_idx], num_iteration=clf.best_iteration)
    getVal[val_idx]+= clf.predict(train_new.iloc[val_idx], num_iteration=clf.best_iteration) / folds.n_splits
    
    
    predictions += clf.predict(test_new, num_iteration=clf.best_iteration) / folds.n_splits
    
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = predictions
sub.to_csv("submission_lgb_final_with_kfold.csv", index=False)

