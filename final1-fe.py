
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


# In[2]:


def add_suffix(df, str):
    res = df.columns.values
    res = res + str
    res[0] = 'id'
    df.columns=res


# # Merge test and A

# In[3]:


train_userinfo = pd.read_csv('../../data/train/train_profile.csv')
train_userinfo.columns=['id','sex','ocp','edu','is_mrge','regc_prop']
train_bankdetail = pd.read_csv('../../data/train/train_bankStatement.csv')
train_bankdetail.columns = ['id','bank_detail_time','trade_type','trade_amt','is_salary']
train_billdetail=pd.read_csv('../../data/train/train_creditBill.csv')
train_billdetail.columns=['id','bank_id','bill_detail_time','pre_act_amt','pre_rpay_amt','cur_act_bal','cr_qumt','rpay_sts']
train_browse=pd.read_csv('../../data/train/train_behaviors.csv')
train_browse.columns=['id','browse_date','week_day','action','child_action1', 'child_action2']
train_overdue = pd.read_csv('../../data/train/train_label.csv')
train_overdue.columns=['id','is_exd']

print(train_userinfo.shape)
print(train_bankdetail.shape)
print(train_browse.shape)
print(train_billdetail.shape)
print(train_overdue.shape)


# In[4]:


A_userinfo = pd.read_csv('../../data/B/test_profile_B.csv')
A_userinfo.columns=['id','sex','ocp','edu','is_mrge','regc_prop']
A_bankdetail = pd.read_csv('../../data/B/test_bankStatement_B.csv')
A_bankdetail.columns = ['id','bank_detail_time','trade_type','trade_amt','is_salary']
A_billdetail=pd.read_csv('../../data/B/test_creditBill_B.csv')
A_billdetail.columns=['id','bank_id','bill_detail_time','pre_act_amt','pre_rpay_amt','cur_act_bal','cr_qumt','rpay_sts']
A_browse=pd.read_csv('../../data/B/test_behaviors_B.csv')
A_browse.columns=['id','browse_date','week_day','action','child_action1', 'child_action2']

print(A_userinfo.shape)
print(A_bankdetail.shape)
print(A_browse.shape)
print(A_billdetail.shape)


# In[5]:


train_userinfo = pd.concat([train_userinfo, A_userinfo],ignore_index=True)
train_bankdetail = pd.concat([train_bankdetail, A_bankdetail],ignore_index=True)
train_billdetail = pd.concat([train_billdetail, A_billdetail],ignore_index=True)
train_browse = pd.concat([train_browse, A_browse],ignore_index=True)

print(train_userinfo.shape)
print(train_bankdetail.shape)
print(train_browse.shape)
print(train_billdetail.shape)
print(train_overdue.shape)


# #  Feature Engineering

# ## //86400

# In[6]:


train_billdetail['bill_detail_time']=train_billdetail['bill_detail_time']//86400
train_bankdetail['bank_detail_time']=train_bankdetail['bank_detail_time']//86400


# In[7]:


#train_bankdetail['bank_detail_time'].value_counts()


# In[8]:


#train_billdetail['bill_detail_time'].value_counts()


# ## 客户信息表

# In[9]:


id = train_userinfo['id']


# In[10]:


train_userinfo.shape


# In[11]:


dataset=train_userinfo.drop(['id'], axis=1)
dataset=pd.get_dummies(dataset,columns=dataset[['sex','ocp','edu','is_mrge','regc_prop']])
dataset.head()


# In[12]:


train_userinfo_feature = pd.concat([id,dataset],axis=1)
train_userinfo_feature.head()


# In[13]:


train_userinfo_feature.shape


# In[14]:


#train_userinfo_feature.to_csv('train_userinfo_feature.csv',index=None)


# 
# ## 银行流水表
# 

# In[15]:


# bank_detail
## 交易类型中0为收入，1为支出。
## 把收入置为1，支出置为-1
train_bankdetail.loc[train_bankdetail['trade_type'] == 1, 'trade_type'] = -1
train_bankdetail.loc[train_bankdetail['trade_type'] == 0, 'trade_type'] = 1

train_bankdetail['in_amount']=train_bankdetail.loc[train_bankdetail['trade_type'] == 1, 'trade_amt']
train_bankdetail['out_amount']=train_bankdetail.loc[train_bankdetail['trade_type'] == -1, 'trade_amt']
train_bankdetail=train_bankdetail.fillna(0)
train_bankdetail['net_amount']=train_bankdetail['in_amount']-train_bankdetail['out_amount']


# In[16]:


train_bankdetail['sal_amt'] = train_bankdetail['trade_amt']*train_bankdetail['is_salary']
train_bankdetail['other_in_amt'] = train_bankdetail['in_amount']-train_bankdetail['sal_amt']


# In[17]:


dataset=train_bankdetail.drop(['trade_type','is_salary'],axis=1)
dataset=dataset.groupby(['id','bank_detail_time'],as_index=False).sum()


# In[18]:


dataset.head()


# In[19]:


dataset_num_all = dataset.groupby('id').agg(
    ['sum', 'mean', 'max', 'median', 'min', 'std', 'skew'])
dataset_num_all.columns = ['_'.join(col).strip()
                             for col in dataset_num_all.columns.values]
dataset_num_all.reset_index(inplace=True)
dataset_num_all.rename(columns=lambda x: x+'_all', inplace=True)
dataset_num_all.rename(columns={ dataset_num_all.columns[0]: 'id' }, inplace = True)


# In[20]:


dataset_num_all.head()


# In[21]:


train_bankdetail_feature=train_bankdetail.groupby(['id'],as_index=False)['trade_type'].count()
train_bankdetail_feature.columns=['id', 'bank_tr_count']


# In[22]:


train_bankdetail_feature=pd.merge(train_bankdetail_feature, dataset_num_all,how='left',on='id')


# In[23]:


train_bankdetail_feature.shape
#train_bankdetail_sum['is_salary'].value_counts()


# In[24]:


train_bankdetail_feature.head()


# In[25]:


#train_bankdetail_feature.columns.value_counts()


# ## 信用卡流水表

# In[26]:


#train_billdetail.columns=['id','bank_id','bill_detail_time','pre_act_amt','pre_rpay_amt','cur_act_bal','cr_qumt','rpay_sts']
#                       用户标识,银行标识,账单时间戳,       上期账单金额,上期还款金额,    本期账单余额, 信用卡额度, 还款状态


# In[97]:


train_billdetail.head(10)


# In[98]:


dataset_bill=train_billdetail[['id','bank_id']]


# In[99]:


#one-hot encoding of bank_id
dataset_feature_bill=pd.get_dummies(dataset_bill,columns=dataset_bill[['bank_id']])
dataset_feature_bill_sum=dataset_feature_bill.groupby(['id'],as_index=False).sum()


# In[100]:


dataset_feature_bill_sum.head()


# In[101]:


dataset_feature_bill_sum.shape


# In[102]:


dataset_feature_bill_sum['bank_id_sum']=dataset_feature_bill_sum.iloc[:,1:14].sum(axis=1)
dataset_feature_bill_sum['bank_id_mean']=dataset_feature_bill_sum.iloc[:,1:14].mean(axis=1)
dataset_feature_bill_sum['bank_id_max']=dataset_feature_bill_sum.iloc[:,1:14].max(axis=1)
dataset_feature_bill_sum['bank_id_std']=dataset_feature_bill_sum.iloc[:,1:14].std(axis=1)



# In[103]:


dataset_feature_bill_sum['bank_id_count']=dataset_feature_bill_sum.iloc[:,1:14].astype(bool).sum(axis=1)


# In[104]:


dataset_feature_bill_sum.head()


# In[105]:


train_billdetail['pre_unpay_amt'] = train_billdetail['pre_act_amt'] - train_billdetail['pre_rpay_amt']
train_billdetail['act_amt_diff'] = train_billdetail['cur_act_bal'] - train_billdetail['pre_act_amt']
train_billdetail['used_amt'] = train_billdetail['cr_qumt'] - train_billdetail['cur_act_bal']
train_billdetail['used_amt_term0'] = train_billdetail['used_amt'] - train_billdetail['pre_unpay_amt']


# In[106]:


train_billdetail_sum=train_billdetail.groupby(['id'],as_index=False).sum()
train_billdetail_mean=train_billdetail.groupby(['id'],as_index=False).mean()
train_billdetail_var=train_billdetail.groupby(['id'],as_index=False).var()
train_billdetail_min=train_billdetail.groupby(['id'],as_index=False).min()
train_billdetail_max=train_billdetail.groupby(['id'],as_index=False).max()
train_billdetail_qt50=train_billdetail.groupby(['id'],as_index=False).median()

add_suffix(train_billdetail_sum,'_sum')
add_suffix(train_billdetail_mean,'_mean')
add_suffix(train_billdetail_var,'_var')
add_suffix(train_billdetail_min,'_min')
add_suffix(train_billdetail_max,'_max')
add_suffix(train_billdetail_qt50,'_median')

train_billdetail_feature=pd.merge(train_billdetail_sum, train_billdetail_mean,how='left',on='id')
train_billdetail_feature=pd.merge(train_billdetail_feature, train_billdetail_var,how='left',on='id')
train_billdetail_feature=pd.merge(train_billdetail_feature, train_billdetail_min,how='left',on='id')
train_billdetail_feature=pd.merge(train_billdetail_feature, train_billdetail_max,how='left',on='id')
train_billdetail_feature=pd.merge(train_billdetail_feature, train_billdetail_qt50,how='left',on='id')
train_billdetail_feature=pd.merge(train_billdetail_feature, dataset_feature_bill_sum,how='left',on='id')


# In[107]:


train_billdetail_feature.shape


# In[108]:


train_billdetail_feature.head()


# In[109]:


bill_feature_act = train_billdetail[['id','bill_detail_time']].groupby('id').agg(
    ['count','nunique'])
bill_feature_act.columns = ['_'.join(col).strip()
                             for col in bill_feature_act.columns.values]
bill_feature_act.reset_index(inplace=True)
bill_feature_act.rename(columns=lambda x: x+'_act', inplace=True)
bill_feature_act.rename(columns={ bill_feature_act.columns[0]: 'id' }, inplace = True)
bill_feature_act['bill_detail_time_ratio_act'] =bill_feature_act['bill_detail_time_count_act']/bill_feature_act['bill_detail_time_nunique_act']


# In[110]:


train_billdetail_feature=pd.merge(train_billdetail_feature, bill_feature_act,how='left',on='id')


# In[111]:


def make_bank(bank_id):
    df = train_billdetail[train_billdetail['bank_id']==bank_id]

    df_sum=df.groupby(['id'],as_index=False).sum()
    df_mean=df.groupby(['id'],as_index=False).mean()
    df_var=df.groupby(['id'],as_index=False).var()
    df_min=df.groupby(['id'],as_index=False).min()
    df_max=df.groupby(['id'],as_index=False).max()
    df_qt50=df.groupby(['id'],as_index=False).median()

    add_suffix(df_sum,'_sum_'+str(bank_id))
    add_suffix(df_mean,'_mean_'+str(bank_id))
    add_suffix(df_var,'_var_'+str(bank_id))
    add_suffix(df_min,'_min_'+str(bank_id))
    add_suffix(df_max,'_max_'+str(bank_id))
    add_suffix(df_qt50,'_median_'+str(bank_id))

    df_feature=pd.merge(df_sum, df_mean,how='left',on='id')
    df_feature=pd.merge(df_feature, df_var,how='left',on='id')
    df_feature=pd.merge(df_feature, df_min,how='left',on='id')
    df_feature=pd.merge(df_feature, df_max,how='left',on='id')
    df_feature=pd.merge(df_feature, df_qt50,how='left',on='id')
    
    return df_feature


# In[112]:


for i in range(0,13):
    train_billdetail_feature = pd.merge(train_billdetail_feature, make_bank(i), how='left', on='id')


# In[113]:


train_billdetail_feature.shape


# In[114]:


#train_billdetail_feature.columns.value_counts()


# ## 浏览记录信息

# ### 日期处理

# In[43]:


def to_days(browse_date_):
    m,d = [int(i) for i in browse_date_.split('-')]
    if m<3:
        return (m-1)*31+d
    elif m<5:
        return (m-1)*31+d-3
    elif m<7:
        return (m-1)*31+d-4
    elif m<10:
        return (m-1)*31+d-5
    elif m<12:
        return (m-1)*31+d-6
    else:
        return (m-1)*31+d-7


# In[44]:


fake_days = ['2-29','2-30','2-31','4-31','6-31','9-31','11-31']
train_browse = train_browse[~train_browse['browse_date'].isin(fake_days)]


# In[45]:


train_browse['browse_date'] = train_browse['browse_date'].apply(to_days)


# In[46]:


days = []
counts = []
nuniq = []
for i in range(1,366):
    d = train_browse[train_browse['browse_date']==i]['week_day'].mode()
    days.append(d)
    c = train_browse[train_browse['browse_date']==i]['week_day'].value_counts()
    counts.append(c)
    n = train_browse[train_browse['browse_date']==i]['week_day'].nunique()
    nuniq.append(n)


# In[47]:


pd.set_option('display.max_rows', 400)  # 设置显示最大行


# In[48]:


counts = pd.DataFrame(counts)
counts.reset_index()


# In[49]:


def get_year(row):
    if row['browse_date'] <= 16:
        base = (row['browse_date']+5) % 7
        if row['week_day']==base:
            return 7
        else:
            return (row['week_day']-base) % 7
    else:
        base = (row['browse_date']+4) % 7
        return (row['week_day']-base+6) % 7


# In[50]:


train_browse['browse_year'] = train_browse.apply(get_year, axis=1)


# In[51]:


train_browse['browse_days'] = train_browse['browse_year']*365 + train_browse['browse_date']


# In[52]:


train_browse.to_pickle('./train_browse.pkl')


# In[53]:


#train_browse = pd.read_pickle('./train_browse.pkl')


# In[54]:


train_browse['browse_days'].max()


# In[55]:


train_browse['browse_days'] = 2572 - train_browse['browse_days']


# In[115]:


browse_feature_days = train_browse[['id','browse_days']].groupby('id').agg(
    ['nunique'])
browse_feature_days.columns = ['_'.join(col).strip()
                             for col in browse_feature_days.columns.values]
browse_feature_days.reset_index(inplace=True)
browse_feature_days.rename(columns=lambda x: x+'_days', inplace=True)
browse_feature_days.rename(columns={ browse_feature_days.columns[0]: 'id' }, inplace = True)
browse_feature_days.head()


# In[116]:


dataset = train_browse[['week_day']]
dataset_weekday=pd.get_dummies(dataset,columns=dataset[['week_day']])
dataset_weekday = pd.concat([train_browse[['id']],dataset_weekday], axis=1)
week_day = dataset_weekday.groupby(['id']).agg(
    ['sum'])
week_day.columns = ['_'.join(col).strip()
                             for col in week_day.columns.values]
week_day.reset_index(inplace=True)
week_day.shape


# In[117]:


week_day['week_day_sum']=week_day.iloc[:,1:week_day.shape[1]].sum(axis=1)


# In[118]:


week_day['week_day_0_per'] = week_day['week_day_0_sum'] / week_day['week_day_sum']
week_day['week_day_1_per'] = week_day['week_day_1_sum'] / week_day['week_day_sum']
week_day['week_day_2_per'] = week_day['week_day_2_sum'] / week_day['week_day_sum']
week_day['week_day_3_per'] = week_day['week_day_3_sum'] / week_day['week_day_sum']
week_day['week_day_4_per'] = week_day['week_day_4_sum'] / week_day['week_day_sum']
week_day['week_day_5_per'] = week_day['week_day_5_sum'] / week_day['week_day_sum']
week_day['week_day_6_per'] = week_day['week_day_6_sum'] / week_day['week_day_sum']


# In[57]:


print(train_browse.shape)
#train_browse = train_browse[train_browse['browse_days'] <= 700]
print(train_browse.shape)


# ### 合并行为和子行为特征

# In[58]:


train_browse['action_all']=train_browse['action'].astype(str) + train_browse['child_action1'].astype(str) + train_browse['child_action2'].astype(str)
train_browse['action_all'].replace(train_browse['action_all'].value_counts()[train_browse['action_all'].value_counts()<1000].index.tolist(), 'others', inplace=True)


# In[59]:


dataset=pd.DataFrame(train_browse['action_all'])
dataset.columns=['action_all']
#one-hot encoding
dataset_feature=pd.get_dummies(dataset,columns=dataset[['action_all']])
dataset_feature = pd.concat([train_browse[['id','browse_days']],dataset_feature], axis=1)
action_all = dataset_feature.groupby(['id','browse_days']).agg(
    ['sum'])
action_all.columns = ['_'.join(col).strip()
                             for col in action_all.columns.values]
action_all.reset_index(inplace=True)
action_all.shape


# In[60]:


action_all['action_all_sum']=action_all.iloc[:,2:action_all.shape[1]].sum(axis=1)


# In[61]:


action_all_sum=action_all.groupby(['id'],as_index=False).sum()
action_all_mean=action_all.groupby(['id'],as_index=False).mean()
action_all_var=action_all.groupby(['id'],as_index=False).var()
action_all_median=action_all.groupby(['id'],as_index=False).median()
action_all_min=action_all.groupby(['id'],as_index=False).min()
action_all_max=action_all.groupby(['id'],as_index=False).max()

add_suffix(action_all_sum,'_sum')
add_suffix(action_all_mean,'_mean')
add_suffix(action_all_var,'_var')
add_suffix(action_all_median,'_median')
add_suffix(action_all_min,'_min')
add_suffix(action_all_max,'_max')


# ### combine browsing features

# In[119]:


train_browse_feature=pd.merge(action_all_sum, action_all_mean,how='left',on='id')
train_browse_feature=pd.merge(train_browse_feature, action_all_var,how='left',on='id')
train_browse_feature=pd.merge(train_browse_feature, action_all_median,how='left',on='id')
train_browse_feature=pd.merge(train_browse_feature, action_all_min,how='left',on='id')
train_browse_feature=pd.merge(train_browse_feature, action_all_max,how='left',on='id')

train_browse_feature=pd.merge(train_browse_feature, week_day,how='left',on='id')
train_browse_feature=pd.merge(train_browse_feature, browse_feature_days,how='left',on='id')


train_browse_feature.shape


# In[120]:


train_browse_feature['brs_bhv_count'] = train_browse.groupby(['id'], as_index=False)['action'].nunique()
train_browse_feature['bhv_1_count'] = train_browse.groupby(['id'], as_index=False)['child_action1'].nunique()
train_browse_feature['bhv_2_count'] = train_browse.groupby(['id'], as_index=False)['child_action2'].nunique()
train_browse_feature['brs_bhv_ratio'] = train_browse_feature['brs_bhv_count'] / train_browse.action.nunique()
train_browse_feature['sub-bhv_1_ratio'] = train_browse_feature['bhv_1_count'] / train_browse.child_action1.nunique()
train_browse_feature['sub-bhv_2_ratio'] = train_browse_feature['bhv_2_count'] / train_browse.child_action2.nunique()


# In[121]:


train_browse_feature.shape


# In[63]:


#train_browse_feature.to_csv('train_browse_feature.csv',index=None)


# ## 汇总

# In[126]:


print(train_userinfo_feature.shape)
print(train_bankdetail_feature.shape)
print(train_billdetail_feature.shape)
print(train_browse_feature.shape)


# In[127]:


# merge to get all features

feature=pd.merge(train_userinfo_feature,train_bankdetail_feature,how='left',on='id')
feature=pd.merge(feature,train_billdetail_feature,how='left',on='id')
feature=pd.merge(feature,train_browse_feature,how='left',on='id')
print(feature.shape)


# In[128]:


feature['browse_days_dur'] = feature['browse_days_max'] - feature['browse_days_min']
feature['bill_detail_time_dur'] = feature['bill_detail_time_max'] - feature['bill_detail_time_min']


# In[130]:


# fill na
feature = feature.replace([np.inf, -np.inf], np.nan)
feature = feature.fillna(-999)

id_a = A_userinfo[['id']]
feature_a=id_a.merge(feature,on='id',how='left')
feature=pd.merge(feature,train_overdue,how='right',on='id')
print(feature.shape)
print(feature_a.shape)


# In[131]:


plt.plot(feature[feature['bill_detail_time_max']>0]['bill_detail_time_max'],'.')


# In[132]:


feature['bill_detail_time_max'].max()


# In[133]:


feature_a['bill_detail_time_max'].min()


# In[134]:


plt.plot(feature_a[feature_a['bill_detail_time_max']>0]['bill_detail_time_max'],'.')


# In[135]:


feature.columns.value_counts().max()


# In[136]:


feature.to_pickle('../feature/final0926.pkl')
feature_a.to_pickle('../feature/final_a0926.pkl')


# In[155]:


feature = pd.read_pickle('../feature/final0916_sel.pkl')
feature_a = pd.read_pickle('../feature/final_a0916_sel.pkl')
#feature_a = pd.read_pickle('../feature/final0916_sel.pkl')


# In[156]:


print(feature.shape)
print(feature_a.shape)


# # fit model

# ## lightgbm

# In[203]:


#feature_train,feature_test=train_test_split(feature,test_size=0.2,random_state=24)


# In[139]:


feature_test = feature[feature['bill_detail_time_max']>44551.0]
feature_train = feature[~feature['id'].isin(feature_test['id'].tolist())]


# In[140]:


print(feature.shape)
print(feature_train.shape)
print(feature_test.shape)


# In[141]:


trn_x=feature_train.drop(['id','is_exd'],axis=1)
trn_y=feature_train['is_exd']
val_x=feature_test.drop(['id','is_exd'],axis=1)
val_y=feature_test['is_exd']


# In[142]:


clf = LGBMClassifier(
    num_threads=12,
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=30,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
    seed=42
)

clf.fit(trn_x, trn_y, 
        eval_set= [(trn_x, trn_y), (val_x, val_y)], 
        eval_metric='auc', verbose=100, early_stopping_rounds=200
       )


# In[145]:


val_pred=clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
y_pred = (val_pred >= 0.5)*1
print(classification_report(val_y, y_pred))
fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
abs(fpr-tpr).max()


# In[112]:


val_pred=clf.predict_proba(val_x, num_iteration=2290)[:, 1]
y_pred = (val_pred >= 0.5)*1
print(classification_report(val_y, y_pred))
fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
abs(fpr-tpr).max()


# In[144]:


# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,val_x.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:50])
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
#plt.savefig('lgbm_importances-01.png')


# In[71]:


INIT_WOODY


# In[81]:


test_pred=clf.predict_proba(feature_a.drop('id',axis=1), num_iteration=clf.best_iteration_)[:, 1]


# In[83]:


result = feature_a[['id']].copy()
result['is_exd'] = test_pred


# In[69]:


fn = '../result/0915b.csv'
result.to_csv(fn, index=False, header=None)


# In[73]:


get_ipython().run_line_magic('predict', '../result/0915b.csv')


# ## xgboost

# In[157]:


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


# In[158]:


print(trn_x.shape)


# In[159]:


get_ipython().run_cell_magic('time', '', "xgb_train=xgb.DMatrix(trn_x,label=trn_y)\nxgb_val=xgb.DMatrix(val_x,label=val_y)\nwatchlist = [(xgb_train, 'train'), (xgb_val, 'valid')]\nmodel=xgb.train(params,xgb_train, 8000, early_stopping_rounds=100, evals = watchlist,verbose_eval=100)\nxgb_pred=xgb.DMatrix(val_x)\npred_test=model.predict(xgb_pred)\ny_pred = (pred_test >= 0.5)*1\nprint(classification_report(val_y, y_pred))")


# In[161]:


fpr,tpr,thres=roc_curve(val_y,pred_test,pos_label=1)
print('ks:', abs(fpr-tpr).max())
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model,height=0.5,ax=ax,max_num_features=64)
plt.show()


# In[93]:





# ## catboost

# In[313]:


clf_cat = CatBoostClassifier(iterations = 4000,
                     learning_rate = 0.02,
                     eval_metric = 'AUC',
                     thread_count = 10,
                     random_seed = 42,
                     od_type = 'Iter',
                     metric_period = 100,
                     od_wait = 100)

clf_cat.fit(trn_x, trn_y, 
        eval_set= [(trn_x, trn_y), (val_x, val_y)], 
        early_stopping_rounds=200, plot=True
       )


# In[314]:


val_pred=clf_cat.predict_proba(val_x)[:, 1]
y_pred = (val_pred >= 0.5)*1
print(classification_report(val_y, y_pred))
fpr,tpr,thres=roc_curve(val_y,val_pred,pos_label=1)
abs(fpr-tpr).max()


# # retrain model and generate A

# ## xgb

# In[162]:


get_ipython().run_cell_magic('time', '', "X=feature.drop(['id','is_exd'],axis=1)\ny=feature['is_exd']\n\nxgb_all=xgb.DMatrix(X,label=y)\nmodel=xgb.train(params,xgb_all, 7000 ,evals = [(xgb_all, 'train')], verbose_eval=100)")


# In[164]:


xgb_a = xgb.DMatrix(feature_a.drop(['id'],axis=1))
pred_a=model.predict(xgb_a)
pred_a


# In[165]:


# save result
result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=pred_a
result.head()


# In[166]:


result.to_csv('../result/0927a.csv',index=None,header=None)


# In[167]:


INIT_WOODY


# In[168]:


get_ipython().run_line_magic('predict', '../result/0927a.csv')


# In[103]:


#result.to_csv('sub1_5439.csv',index=None,header=None)
#model.save_model('model_sub1_5439.model')


# ## lightgbm

# In[153]:


clf = LGBMClassifier(
    num_threads=10,
    n_estimators=2300,
    learning_rate=0.03,
    num_leaves=30,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
    seed=42
)

clf.fit(X, y, 
        eval_set= [(X, y)], 
        eval_metric='auc', verbose=100
       )


# In[157]:


test_pred=clf.predict_proba(feature_a.drop('id',axis=1), num_iteration=1300)[:, 1]


# In[158]:


test_pred


# In[159]:


# save result
result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=test_pred
result.head()


# In[160]:


INIT_WOODY


# In[161]:


result.to_csv('../result/0916b.csv',index=None,header=None)


# In[162]:


get_ipython().run_line_magic('predict', '../result/0916b.csv')


# ## catboost

# In[112]:


clf_cat = CatBoostClassifier(iterations = 3000,
                     learning_rate = 0.02,
                     depth = 8,
                     eval_metric = 'AUC',
                     thread_count = 10,
                     random_seed = 42,
                     od_type = 'Iter',
                     metric_period = 100,
                     od_wait = 100)

clf_cat.fit(trn_x, trn_y, 
        eval_set= [(trn_x, trn_y), (val_x, val_y)], 
        early_stopping_rounds=200, plot=True
       )


# In[133]:


test_pred=clf_cat.predict_proba(feature_a.drop('id',axis=1))[:, 1]


# In[135]:


# save result
result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=test_pred
result.head()


# ## bagging

# In[115]:


ypred_list = []


# In[116]:


for seed in [1000, 8876, 27, 29581]:
    params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':7,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'min_child_weight':2,
        'eta': 0.01,
        'seed':seed,
        'nthread':12,
        'silent':1}
    
    plst = params.items()
    
    bst = xgb.train(plst, xgb_all, 1000, verbose_eval=10)
    
    ypred_list.append(bst.predict(xgb_a))
    


# In[117]:


for seed in [1000, 27]:
    params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':7,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'min_child_weight':2,
        'eta': 0.01,
        'seed':seed,
        'nthread':12,
        'silent':1}
    
    plst = params.items()
    
    bst = xgb.train(plst, xgb_all, 2000, verbose_eval=10)
    
    ypred_list.append(bst.predict(xgb_a))
    


# In[118]:


for seed in [1000, 27]:
    params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':8,
        'lambda':10,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'min_child_weight':2,
        'eta': 0.01,
        'seed':seed,
        'nthread':12,
        'silent':1}
    
    plst = params.items()
    
    bst = xgb.train(plst, xgb_all, 1000, verbose_eval=10)
    
    ypred_list.append(bst.predict(xgb_a))


# In[119]:


ypred_list


# In[120]:


pred = np.mean(np.array(ypred_list), axis=0)


# In[121]:


sum((pred >= 0.5)*1)


# In[122]:


result=pd.DataFrame(index=None,columns=['id', 'pred'])
result['id']=feature_a['id']
result['pred']=pred
result.head()


# In[123]:


result.to_csv('temp2.csv',index=None,header=None)


# In[124]:


#result.to_csv('sub3_5471.csv',index=None,header=None)

