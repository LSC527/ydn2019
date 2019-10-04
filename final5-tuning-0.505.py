
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['font.serif'] = ['DejaVu Sans']

from xgboost import XGBClassifier
from xgboost import plot_importance
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_curve,roc_curve,auc,classification_report,roc_auc_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ## 手动调参

# In[2]:


feature = pd.read_pickle(
    '../feature/final0926_sel.pkl'
)
feature_a = pd.read_pickle(
    '../feature/final_a0926_sel.pkl'
)


# In[4]:


feature_test = feature[feature['bill_detail_time_max']>44460.0]
feature_train = feature[~feature['id'].isin(feature_test['id'].tolist())]
trn2_x=feature_train.drop(['id','is_exd'],axis=1)
trn2_y=feature_train['is_exd']
print(trn2_x.shape)
feature_val = feature_train[feature_train['bill_detail_time_max']>44350.0]
feature_train = feature_train[~feature_train['id'].isin(feature_val['id'].tolist())]
print(feature.shape)
print(feature_train.shape)
print(feature_test.shape)
print(feature_val.shape)
trn_x=feature_train.drop(['id','is_exd'],axis=1)
trn_y=feature_train['is_exd']
val_x=feature_val.drop(['id','is_exd'],axis=1)
val_y=feature_val['is_exd']
tst_x=feature_test.drop(['id','is_exd'],axis=1)
tst_y=feature_test['is_exd']


# In[6]:


#%%capture output
for depth in [3]:
    for min_child_weight in [3,5,7,9]:
        for gamma in [0.2]:
            for subsample in [0.8]:
                for colsample_bytree in [0.8]:
                    for afa in [0]:
                        for lam in [5]:
                            params={'booster':'gbtree',
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'max_depth':depth,
                            'alpha':afa,
                            'lambda':lam,
                            'gamma': gamma,
                            'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                            'min_child_weight':min_child_weight,
                            'eta': 0.1,
                            'seed':76,
                            'nthread':24,
                            'silent':1}

                            xgb_train=xgb.DMatrix(trn_x,label=trn_y)
                            xgb_val=xgb.DMatrix(val_x,label=val_y)
                            watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score1 = model.best_score

                            xgb_train=xgb.DMatrix(trn2_x,label=trn2_y)
                            xgb_tst=xgb.DMatrix(tst_x,label=tst_y)
                            watchlist = [(xgb_train, 'train'),(xgb_tst, 'test')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score2 = model.best_score

                            print('********************************************************')
                            print(depth, min_child_weight, gamma, subsample, colsample_bytree, afa, lam)
                            print(score1, score2, (score1+score2)/2)
                            print('********************************************************')

                            #xgb_pred=xgb.DMatrix(val_x)
                            #pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)

                            #fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)
                            #print('********************************************************')
                            #print(depth, colsample_bytree, subsample, min_child_weight,gamma)
                            #print('ks:', abs(fpr-tpr).max())
                            #print('********************************************************')

                            #ks = []
                            #rr = range(model.best_ntree_limit-300,model.best_ntree_limit,2)
                            #for i in rr:
                            #    val_pred = model.predict(xgb_pred,ntree_limit=i)
                            #    fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
                            #    ks.append(abs(fpr-tpr).max())
                            #plt.plot(rr, ks)
                            #plt.show()


# In[8]:


#%%capture output
for depth in [3]:
    for min_child_weight in [1,2,3,4]:
        for gamma in [0.2]:
            for subsample in [0.8]:
                for colsample_bytree in [0.8]:
                    for afa in [0]:
                        for lam in [5]:
                            params={'booster':'gbtree',
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'max_depth':depth,
                            'alpha':afa,
                            'lambda':lam,
                            'gamma': gamma,
                            'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                            'min_child_weight':min_child_weight,
                            'eta': 0.1,
                            'seed':76,
                            'nthread':24,
                            'silent':1}

                            xgb_train=xgb.DMatrix(trn_x,label=trn_y)
                            xgb_val=xgb.DMatrix(val_x,label=val_y)
                            watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score1 = model.best_score

                            xgb_train=xgb.DMatrix(trn2_x,label=trn2_y)
                            xgb_tst=xgb.DMatrix(tst_x,label=tst_y)
                            watchlist = [(xgb_train, 'train'),(xgb_tst, 'test')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score2 = model.best_score

                            print('********************************************************')
                            print(depth, min_child_weight, gamma, subsample, colsample_bytree, afa, lam)
                            print(score1, score2, (score1+score2)/2)
                            print('********************************************************')


# In[9]:


#%%capture output
for depth in [3]:
    for min_child_weight in [3]:
        for gamma in [0,0.1,0.2,0.5,1]:
            for subsample in [0.8]:
                for colsample_bytree in [0.8]:
                    for afa in [0]:
                        for lam in [5]:
                            params={'booster':'gbtree',
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'max_depth':depth,
                            'alpha':afa,
                            'lambda':lam,
                            'gamma': gamma,
                            'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                            'min_child_weight':min_child_weight,
                            'eta': 0.1,
                            'seed':76,
                            'nthread':24,
                            'silent':1}

                            xgb_train=xgb.DMatrix(trn_x,label=trn_y)
                            xgb_val=xgb.DMatrix(val_x,label=val_y)
                            watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score1 = model.best_score

                            xgb_train=xgb.DMatrix(trn2_x,label=trn2_y)
                            xgb_tst=xgb.DMatrix(tst_x,label=tst_y)
                            watchlist = [(xgb_train, 'train'),(xgb_tst, 'test')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score2 = model.best_score

                            print('********************************************************')
                            print(depth, min_child_weight, gamma, subsample, colsample_bytree, afa, lam)
                            print(score1, score2, (score1+score2)/2)
                            print('********************************************************')


# In[10]:


#%%capture output
for depth in [3]:
    for min_child_weight in [3]:
        for gamma in [0.1]:
            for subsample in [0.7,0.8,0.9]:
                for colsample_bytree in [0.7,0.8,0.9]:
                    for afa in [0]:
                        for lam in [5]:
                            params={'booster':'gbtree',
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'max_depth':depth,
                            'alpha':afa,
                            'lambda':lam,
                            'gamma': gamma,
                            'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                            'min_child_weight':min_child_weight,
                            'eta': 0.1,
                            'seed':76,
                            'nthread':24,
                            'silent':1}

                            xgb_train=xgb.DMatrix(trn_x,label=trn_y)
                            xgb_val=xgb.DMatrix(val_x,label=val_y)
                            watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score1 = model.best_score

                            xgb_train=xgb.DMatrix(trn2_x,label=trn2_y)
                            xgb_tst=xgb.DMatrix(tst_x,label=tst_y)
                            watchlist = [(xgb_train, 'train'),(xgb_tst, 'test')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score2 = model.best_score

                            print('********************************************************')
                            print(depth, min_child_weight, gamma, subsample, colsample_bytree, afa, lam)
                            print(score1, score2, (score1+score2)/2)
                            print('********************************************************')


# In[ ]:


3 3 0.1 0.9 0.9 0 5
0.805797 0.824666 0.8152315


# In[11]:


#%%capture output
for depth in [3]:
    for min_child_weight in [3]:
        for gamma in [0.1]:
            for subsample in [0.85,0.9,0.95]:
                for colsample_bytree in [0.85,0.9,0.95]:
                    for afa in [0]:
                        for lam in [5]:
                            params={'booster':'gbtree',
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc',
                            'max_depth':depth,
                            'alpha':afa,
                            'lambda':lam,
                            'gamma': gamma,
                            'subsample':subsample,
                            'colsample_bytree':colsample_bytree,
                            'min_child_weight':min_child_weight,
                            'eta': 0.1,
                            'seed':76,
                            'nthread':24,
                            'silent':1}

                            xgb_train=xgb.DMatrix(trn_x,label=trn_y)
                            xgb_val=xgb.DMatrix(val_x,label=val_y)
                            watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score1 = model.best_score

                            xgb_train=xgb.DMatrix(trn2_x,label=trn2_y)
                            xgb_tst=xgb.DMatrix(tst_x,label=tst_y)
                            watchlist = [(xgb_train, 'train'),(xgb_tst, 'test')]
                            model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
                            score2 = model.best_score

                            print('********************************************************')
                            print(depth, min_child_weight, gamma, subsample, colsample_bytree, afa, lam)
                            print(score1, score2, (score1+score2)/2)
                            print('********************************************************')


# In[ ]:


********************************************************
3 3 0.1 0.9 0.85 0 5
0.805761 0.82511 0.8154355
********************************************************
depth, min_child_weight, gamma, subsample, colsample_bytree, afa, lam


# In[6]:


output.show()


# ## eta=0.01

# In[8]:


params={'booster':'gbtree',
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth':4,
'gamma':0.2,
'lambda':5,
'subsample':0.8,
'colsample_bytree':0.8,
'min_child_weight':7,
'eta': 0.01,
'seed':1000,
'nthread':20,
'silent':1}


# In[9]:


get_ipython().run_cell_magic('time', '', "xgb_train=xgb.DMatrix(trn2_x,label=trn2_y)\nxgb_tst=xgb.DMatrix(tst_x,label=tst_y)\nwatchlist = [(xgb_train, 'train'), (xgb_tst, 'test')]\nmodel=xgb.train(params,xgb_train, 10000, early_stopping_rounds=300, evals = watchlist,verbose_eval=100)\nxgb_pred=xgb.DMatrix(tst_x)\npred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)\nfpr,tpr,thres=roc_curve(tst_y,pred_test,pos_label=1)\nprint('ks:', abs(fpr-tpr).max())")


# In[ ]:


ks = []
rr = range(model.best_ntree_limit-5000,model.best_ntree_limit+300,50)
for i in rr:
    val_pred = model.predict(xgb_pred,ntree_limit=i)
    fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
    ks.append(abs(fpr-tpr).max())
plt.plot(rr, ks)
plt.show()


# ## retrain

# In[33]:


params_tuned={'booster':'gbtree',
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth':4,
'min_child_weight':7,
'gamma':0.2
'subsample':0.8,
'colsample_bytree':0.8,
'alpha':0
'lambda':5,
'eta': 0.01,
'seed':1000,
'nthread':10,
'silent':1}


# In[ ]:


get_ipython().run_cell_magic('time', '', "X=feature.drop(['id','is_exd'],axis=1)\ny=feature['is_exd']\n\nxgb_all=xgb.DMatrix(X,label=y)\nmodel_all=xgb.train(params_tuned,xgb_all, 3900 ,evals = [(xgb_all, 'train')], verbose_eval=100)")


# In[18]:


get_ipython().run_cell_magic('time', '', "X=feature.drop(['id','is_exd'],axis=1)\ny=feature['is_exd']\n\nxgb_all=xgb.DMatrix(X,label=y)\nmodel_all=xgb.train(params_tuned,xgb_all, 3900 ,evals = [(xgb_all, 'train')], verbose_eval=100)")


# In[22]:


xgb_a = xgb.DMatrix(feature_a.drop(['id'],axis=1))
pred_a=model_all.predict(xgb_a)
# save result
result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=pred_a
result.head()
result.to_csv('../result/0920c.csv',index=None,header=None)


# In[25]:


INIT_WOODY


# In[26]:


get_ipython().run_line_magic('predict', '../result/0920c.csv')


# ## retrain with na filled

# In[27]:


feature = feature.replace([np.inf, -np.inf], np.nan)
feature = feature.fillna(-999)
feature_a = feature_a.replace([np.inf, -np.inf], np.nan)
feature_a = feature_a.fillna(-999)


# In[39]:


get_ipython().run_cell_magic('time', '', "X=feature.drop(['id','is_exd'],axis=1)\ny=feature['is_exd']\n\nxgb_all=xgb.DMatrix(X,label=y)\nmodel_all=xgb.train(params_tuned,xgb_all, 3900 ,evals = [(xgb_all, 'train')], verbose_eval=100)")


# In[42]:


xgb_a = xgb.DMatrix(feature_a.drop(['id'],axis=1))
pred_a=model_all.predict(xgb_a,ntree_limit=3743)
# save result
result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=pred_a
result.to_csv('../result/0921b.csv',index=None,header=None)
result.head()


# In[41]:


result.head()


# In[36]:


result.head()


# In[43]:


INIT_WOODY


# In[44]:


get_ipython().run_line_magic('predict', '../result/0921b.csv')

