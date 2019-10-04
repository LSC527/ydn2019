
# coding: utf-8

# ### Feature selecture using target permutation
# 
# The notebook uses a procedure described in [this article]( https://academic.oup.com/bioinformatics/article/26/10/1340/193348).
# 
# Feature selection process using target permutation tests actual importance significance against the distribution of feature importances when fitted to noise (shuffled target).
# 
# The notebook implements the following steps  :
#  - Create the null importances distributions : these are created fitting the model over several runs on a shuffled version of the target. This shows how the model can make sense of a feature irrespective of the target.
#  - Fit the model on the original target and gather the feature importances. This gives us a benchmark whose significance can be tested against the Null Importances Distribution
#  - for each feature test the actual importance:
#     - Compute the probabability of the actual importance wrt the null distribution. I will use a very simple estimation using occurences while the article proposes to fit known distribution to the gathered data. In fact here I'll compute 1 - the proba so that things are in the right order.
#     - Simply compare the actual importance to the mean and max of the null importances. This will give sort of a feature importance that allows to see major features in the dataset. Indeed the previous method may give us lots of ones.
# 
# For processing time reasons, the notebook will only cover application_train.csv but you can extend it as you wish.
# 

# ### Import a few packages

# In[1]:


import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter('ignore', UserWarning)

import gc
gc.enable()


# ### Read application_train
# 
# Read data and take care of categorical features

# In[2]:


# Load data
df_feature = pd.read_pickle(
    '../feature/feature0916.pkl'
)

df_feature.shape


# ### Create a scoring function
# 
# Coring function uses LightGBM in RandomForest mode fitted on the full dataset 

# In[5]:


def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data if f not in ['id', 'is_exd']]
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = data['is_exd'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['is_exd'].copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 20
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))
    
    return imp_df


# ### Build the benchmark for feature importance
# 
# ![](http://)The original paper does not talk about this but I think it makes sense to have a distribution of actual importances as well

# In[ ]:


# Seed the unexpected randomness of this world
np.random.seed(123)
# Get the actual importance, i.e. without shuffling
actual_imp_df = get_feature_importances(data=df_feature, shuffle=False)


# In[22]:


actual_imp_df.head(10)


# ### Build Null Importances distribution

# In[ ]:


null_imp_df = pd.DataFrame()
nb_runs = 80
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=df_feature, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)


# In[23]:


null_imp_df.shape


# In[24]:


actual_imp_df.shape


# ### Display distribution examples
# 
# A few plots are better than any words

# In[ ]:


def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
        


# In[25]:


display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='brs_bhv_ratio')


# From the above plot I believe the power of the exposed feature selection method is demonstrated. In particular it is well known that :
#  - Any feature sufficient variance can be used and made sense of by tree models. You can always find splits that help scoring better
#  - Correlated features have decaying importances once one of them is used by the model. The chosen feature will have strong importance and its correlated suite will have decaying importances
#  
#  The current method allows to :
#   - Drop high variance features if they are not really related to the target
#   - Remove the decaying factor on correlated features, showing their real importance (or unbiased importance)
# 

# ### Score features
# 
# There are several ways to score features : 
#  - Compute the number of samples in the actual importances that are away from the null importances recorded distribution.
#  - Compute ratios like Actual / Null Max, Actual  / Null Mean,  Actual Mean / Null Max
#  
# In a first step I will use the log actual feature importance divided by the 75 percentile of null distribution.

# In[13]:


feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()


# In[14]:


scores_df.shape


# ### Save data

# In[15]:


null_imp_df.to_csv('null_importances_distribution_rf.csv')
actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')


# ### Check the impact of removing uncorrelated features
# 
# Here I'll use a different metric to asses correlation to the target

# In[16]:


correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))

corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

fig = plt.figure(figsize=(16, 16))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='split_score', y='feature', data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70], ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
fig.subplots_adjust(top=0.93)


# ### Score feature removal for different thresholds

# In[26]:


def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
    # Fit LightGBM 
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': .1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 13,
        'n_jobs': 6,
        'min_split_gain': .00001,
        'reg_alpha': .00001,
        'reg_lambda': .00001,
        'metric': 'auc'
    }
    
    # Fit the model
    hist = lgb.cv(
        params=lgb_params, 
        train_set=dtrain, 
        num_boost_round=2000,
        #categorical_feature=cat_feats,
        nfold=3,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=17
    )
    # Return the last mean / std values 
    return hist['auc-mean'][-1], hist['auc-stdv'][-1]

# features = [f for f in data.columns if f not in ['SK_ID_CURR', 'TARGET']]
# score_feature_selection(df=data[features], train_features=features, target=data['TARGET'])

for threshold in [0, 10, 20, 30 , 40, 50]:
    split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
    #split_cat_feats = [_f for _f, _score, _ in correlation_scores if (_score >= threshold) & (_f in categorical_feats)]
    gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
    #gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= threshold) & (_f in categorical_feats)]
                                                                                             
    print('Results for threshold %3d' % threshold)
    split_results = score_feature_selection(df=df_feature, train_features=split_feats, target=df_feature['is_exd'])
    print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
    gain_results = score_feature_selection(df=df_feature, train_features=gain_feats, target=df_feature['is_exd'])
    print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))


# ### validate without cv

# In[27]:


feature_test = df_feature[df_feature['bill_detail_time_max']>44551.0]
feature_train = df_feature[~df_feature['id'].isin(feature_test['id'].tolist())]
print(df_feature.shape)
print(feature_train.shape)
print(feature_test.shape)
trn_x=feature_train.drop(['id','is_exd'],axis=1)
trn_y=feature_train['is_exd']
val_x=feature_test.drop(['id','is_exd'],axis=1)
val_y=feature_test['is_exd']


# In[28]:


params={'booster':'gbtree',
'objective': 'binary:logistic',
'eval_metric': 'auc',
'max_depth':6,
'lambda':10,
'subsample':0.7,
'colsample_bytree':0.8,
'min_child_weight':3,
'eta': 0.1,
'seed':1000,
'nthread':20,
'silent':1}


# In[20]:


xgb_train=xgb.DMatrix(trn_x,label=trn_y)
xgb_val=xgb.DMatrix(val_x,label=val_y)
watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
xgb_pred=xgb.DMatrix(val_x)
pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)
fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)
print('all feat ks:', abs(fpr-tpr).max())

ks = []
rr = range(model.best_ntree_limit-300,model.best_ntree_limit,2)
for i in rr:
    val_pred = model.predict(xgb_pred,ntree_limit=i)
    fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
    ks.append(abs(fpr-tpr).max())
plt.plot(rr, ks)
plt.show()


# In[29]:


for threshold in [10, 20, 30 , 40, 50]:
    split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
    gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
                                                                                             
    print('Results for threshold %3d' % threshold)
    print(len(split_feats),len(gain_feats))
    
    xgb_train=xgb.DMatrix(trn_x[split_feats],label=trn_y)
    xgb_val=xgb.DMatrix(val_x[split_feats],label=val_y)
    watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
    model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
    xgb_pred=xgb.DMatrix(val_x[split_feats])
    pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)
    fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)
    print('********************************************************')
    print('Results for threshold %3d' % threshold)
    print('split feat ks:', abs(fpr-tpr).max())
    print('********************************************************')
    ks = []
    rr = range(model.best_ntree_limit-300,model.best_ntree_limit,2)
    for i in rr:
        val_pred = model.predict(xgb_pred,ntree_limit=i)
        fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
        ks.append(abs(fpr-tpr).max())
    plt.plot(rr, ks)
    plt.show()
    
    xgb_train=xgb.DMatrix(trn_x[gain_feats],label=trn_y)
    xgb_val=xgb.DMatrix(val_x[gain_feats],label=val_y)
    watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
    model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)
    xgb_pred=xgb.DMatrix(val_x[gain_feats])
    pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)
    fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)
    print('********************************************************')
    print('Results for threshold %3d' % threshold)
    print('gain feat ks:', abs(fpr-tpr).max())
    print('********************************************************')
    ks = []
    rr = range(model.best_ntree_limit-300,model.best_ntree_limit,2)
    for i in rr:
        val_pred = model.predict(xgb_pred,ntree_limit=i)
        fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
        ks.append(abs(fpr-tpr).max())
    plt.plot(rr, ks)
    plt.show()


# ### validate with xgb_ks

# In[ ]:


def xgb_ks(preds,dtrain):
    labels = dtrain.get_label()
    fpr,tpr,thres = roc_curve(labels,preds,pos_label=1)
    ks = max(tpr-fpr)
    return 'ks',ks


# In[ ]:


params={'booster':'gbtree',
'objective': 'binary:logistic',
#'eval_metric': 'auc',
'max_depth':6,
'lambda':10,
'subsample':0.7,
'colsample_bytree':0.7,
'min_child_weight':5,
'eta': 0.1,
'seed':1000,
'nthread':12,
'silent':1}


# In[ ]:


xgb_train=xgb.DMatrix(trn_x,label=trn_y)
xgb_val=xgb.DMatrix(val_x,label=val_y)
watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]
model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=500, evals = watchlist,verbose_eval=100,feval=xgb_ks, maximize=True)
xgb_pred=xgb.DMatrix(val_x)


# In[ ]:


pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)
fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)
print('all feat ks:', abs(fpr-tpr).max())


# In[ ]:


get_ipython().run_cell_magic('capture', 'output', "#for threshold in [10, 20]:\nfor threshold in [10, 20, 30 , 40, 50]:\n    split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]\n    gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]\n                                                                                             \n    print('Results for threshold %3d' % threshold)\n    print(len(split_feats),len(gain_feats))\n    \n    xgb_train=xgb.DMatrix(trn_x[split_feats],label=trn_y)\n    xgb_val=xgb.DMatrix(val_x[split_feats],label=val_y)\n    watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]\n    model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=500, evals = watchlist,verbose_eval=100,feval=xgb_ks, maximize=True)\n    xgb_pred=xgb.DMatrix(val_x[split_feats])\n    pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)\n    fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)\n    \n    print('********************************************************')\n    print('Results for threshold %3d' % threshold)\n    print('split feat ks:', abs(fpr-tpr).max())\n    print('********************************************************')\n    \n    xgb_train=xgb.DMatrix(trn_x[gain_feats],label=trn_y)\n    xgb_val=xgb.DMatrix(val_x[gain_feats],label=val_y)\n    watchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]\n    model=xgb.train(params,xgb_train, 4000, early_stopping_rounds=500, evals = watchlist,verbose_eval=100,feval=xgb_ks, maximize=True)\n    xgb_pred=xgb.DMatrix(val_x[gain_feats])\n    pred_test=model.predict(xgb_pred,ntree_limit=model.best_ntree_limit)\n    fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)\n    print('********************************************************')\n    print('Results for threshold %3d' % threshold)\n    print('gain feat ks:', abs(fpr-tpr).max())\n    print('********************************************************')\n    \n    ")


# In[ ]:


output.show()


# ### save feature list

# In[29]:


feats = [_f for _f, _score, _ in correlation_scores if _score >= 40]


# In[30]:


feature = pd.read_pickle('../feature/feature0924.pkl')
feature_a = pd.read_pickle('../feature/feature_a0924.pkl')
sel_feature = pd.concat([feature[['id','is_exd']],feature[feats]], axis=1)
sel_feature_a = pd.concat([feature_a[['id']],feature_a[feats]], axis=1)
sel_feature.to_pickle('../feature/feature0924_sel.pkl')
sel_feature_a.to_pickle('../feature/feature_a0924_sel.pkl')


# ### retrain on full set

# In[ ]:


gain_feats = [_f for _f, _score, _ in correlation_scores if _score >= 10]


# In[62]:


get_ipython().run_cell_magic('time', '', "X=df_feature.drop(['id','is_exd'],axis=1)\ny=df_feature['is_exd']\n\nxgb_all=xgb.DMatrix(X[gain_feats],label=y)\nmodel=xgb.train(params,xgb_all, 3400 ,evals = [(xgb_all, 'train')], verbose_eval=100)")


# In[63]:


feature_a = pd.read_pickle('../feature/feature_a0916.pkl')
xgb_a = xgb.DMatrix(feature_a[gain_feats])
pred_a=model.predict(xgb_a)
pred_a
# save result
result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=pred_a
result.head()


# In[64]:


result.to_csv('../result/0918a.csv',index=None,header=None)


# In[65]:


INIT_WOODY


# In[66]:


get_ipython().run_line_magic('predict', '../result/0918a.csv')

