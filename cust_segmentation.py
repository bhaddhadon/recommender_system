# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np 
import re
import openpyxl
import datetime
import pickle
from functools import reduce
import os
import formatting

# %% [markdown]
# ## Read CSV

# %%
df_raw = pd.read_csv('Data.csv')
df_raw.columns = [i.lower() for i in df_raw.columns.values]

# %% [markdown]
# ## Data formatting

# %%
## for classification model 
df1 = df_raw.copy()

date_cols = ['issue_date','st_date']
desc_cols = ['pol_st','product_category','cover_type','prod_type','relationship','owner_gender','region_dummy']

for col in date_cols:
    df1[col] = df1[col].apply(lambda x: None if pd.isnull(x) else datetime.datetime.strptime(x,'%m/%d/%Y'))

for col in desc_cols:
    df1[col] = df1[col].apply(lambda x: str(x).lower())

print(f"dtypes:{df1.dtypes}")

df1['buyer_id'] = np.where(df1['buyer_id'].isnull(),df1['insured_id'],df1['buyer_id'])
df1['buyer_buying_for'] = np.where(df1['relationship']=='self','self','dependent')

### remove policies bought by 'EMPLOYER'

print(f"rows bf removal:{df1.shape[0]}")
df1 = df1.iloc[np.where(df1['relationship'].str.upper()!='EMPLOYER')]
print(f"rows aft removing employer purchase:{df1.shape[0]}")


### remove policies not underwritten

df1 = df1.iloc[np.where(df1['last_uw_dec'].str.strip()=='OK TO ISSUE')]
print(f"rows aft removing policies not underwritten:{df1.shape[0]}")

### remove policies with missing issue_date

df1 = df1.iloc[np.where(df1['issue_date'].notnull())]
print(f"rows aft removing policie w/o issue date:{df1.shape[0]}")

### alter st_date in case where pol_st is not inforce and st_date is ealier than issue_date
# change from original st_date value with issue_date
df1['st_date'] = np.where((df1['st_date']<df1['issue_date'])&(df1['pol_st']!='inforce'),
                          df1['issue_date'],
                          df1['st_date'])

### remove duplicate rows (same policy_ref_dummy, cover_code, cover_type, insured_id)

subset = ['policy_ref_dummy', 'cover_code', 'cover_type', 'insured_id']
df1.drop_duplicates(subset=subset,keep='first',inplace=True)
print(f"rows aft deduplicated:{df1.shape[0]}")


### flg_inforce

df1['flg_inforce'] = np.where(df1['pol_st'].str.strip()=='inforce',1.0,0.0)

### anp

df1['anp'] = df1[['premium','pay_method']].apply(lambda row: formatting.calculate_anp(row['premium'],row['pay_method']),axis=1)

### insured_amt

df1['insured_amt'] = df1['insured_amt'].apply(lambda x: 0.0 
if (pd.isnull(x) or (str(x).replace('.','').replace(',','').isdigit()==False))
else float(str(x).replace(',','')))

### age_grp

df1['owner_age_grp'] = df1['owner_age'].apply(formatting.age_grp)

### impute gender

gender_mode = df1['owner_gender'].mode()
df1['owner_gender'] = np.where(df1['owner_gender'].isnull(),gender_mode,df1['owner_gender'])

### group occupation class further

df1['owner_occupation_grp'] = df1['owner_occupation_class'].apply(formatting.occupation_grp)

### impute occupation_grp by mode of each owner_age_grp

df_agegrp_occ_mode = df1.groupby(by=['owner_age_grp','owner_occupation_grp'],as_index=False)['policy_ref_dummy'].count().rename(columns={'policy_ref_dummy':'count'})
df_agegrp_occ_mode = df_agegrp_occ_mode.sort_values(by=['owner_age_grp','count'],ascending=[1,0])
df_agegrp_occ_mode.drop_duplicates(subset='owner_age_grp',keep='first',inplace=True)
occ_mode_dict = df_agegrp_occ_mode.set_index('owner_age_grp').drop(columns='count',axis=1).to_dict('index')
df1['owner_occupation_grp'] = df1.apply(lambda row: occ_mode_dict[row['owner_age_grp']]['owner_occupation_grp']
if pd.isnull(row['owner_occupation_grp']) else row['owner_occupation_grp'],axis=1)

### create category dummies, prod_type dummies, rider dummies and dependent dummies

df1 = df1.sort_values(by=['policy_ref_dummy','cover_type','buyer_buying_for'],ascending=[1,1,0]).reset_index(drop=True)

df_cat_dummies = pd.get_dummies(df1['product_category'])
df_ul_dummies = pd.get_dummies(df1['prod_type'])
df_cover_dummies = pd.get_dummies(df1['cover_type'])
df_buyingfor_dummies = pd.get_dummies(df1['buyer_buying_for'])

dfs = [df1, df_cat_dummies, df_ul_dummies, df_cover_dummies, df_buyingfor_dummies]
df2 = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,how='left'), dfs)
print(f"rows aft merged with dummies:{df2.shape[0]}")

### create insured_amt by cat, by ul-nul and by self-dependent

cat_cols = ['health','investment','protection','retirement','savings','ul','nul','self','dependent']
cat_insured = [i+'_insured_amt' for i in cat_cols]
cat_anp = [i+'_anp' for i in cat_cols]

for i,col in enumerate(cat_cols):
    df2[cat_insured[i]] = df2.apply(lambda row: row[col]*row['insured_amt'],axis=1)
    df2[cat_anp[i]] = df2.apply(lambda row: row[col]*row['anp'],axis=1)

### grouping from coverage-insured_id level to policy level
## checked that all coverages under policy share the same issue date, pay_mode and status

dim_vars = ['buyer_id','policy_ref_dummy','issue_date','st_date','pol_st','pay_mode',
            'owner_age_grp','owner_gender','owner_occupation_grp','region_dummy','flg_inforce']

def pol_agg(x):
    names = {
        'insured_amt':x['insured_amt'].sum(),
        'anp':x['anp'].sum(),
        'basic':x['basic'].sum(),
        'rider':x['rider'].sum()}
    for col in cat_cols:
        names[col] = x[col].max()
    for col in cat_insured:
        names[col] = x[col].sum()
    for col in cat_anp:
        names[col] = x[col].sum()
    return pd.Series(names, index=[i for i in names.keys()])

df3 = df2.groupby(by=dim_vars,as_index=False).apply(pol_agg)

print(f"rows aft grouped into policy level:{df3.shape[0]}")

### add policy seq

df3 = df3.sort_values(by=['buyer_id','issue_date'],ascending=[1,1]).reset_index(drop=True)
df3['policy_seq'] = df3.groupby(by=['buyer_id'])['policy_ref_dummy'].cumcount()+1

### add column list of insured_amt and anp of each type

df3['insured_amt_dummies'] = df3.apply(lambda row: [row[i] for i in cat_insured],axis=1)
df3['anp_dummies'] = df3.apply(lambda row: [row[i] for i in cat_anp],axis=1)

### list of flg_inforce, st_date, insured_amt and anp for each buyer_id

df_list_flg_inforce = df3.groupby(by='buyer_id')['flg_inforce'].apply(list).reset_index().rename(columns={'flg_inforce':'list_flg_inforce'})
df_list_st_date = df3.groupby(by='buyer_id')['st_date'].apply(list).reset_index().rename(columns={'st_date':'list_st_date'})
df_list_insured_amt = df3.groupby(by='buyer_id')['insured_amt_dummies'].apply(list).reset_index().rename(columns={'insured_amt_dummies':'list_insured_amt'})
df_list_anp = df3.groupby(by='buyer_id')['anp_dummies'].apply(list).reset_index().rename(columns={'anp_dummies':'list_anp'})

dfs = [df3, df_list_flg_inforce, df_list_st_date, df_list_insured_amt, df_list_anp]
df = reduce(lambda left,right: pd.merge(left,right,on='buyer_id',how='left'), dfs)
print(f"rows aft merged with buyer_id's level columns:{df.shape[0]}")

accum_insured = [i+'_acc' for i in cat_insured]
accum_anp = [i+'_acc' for i in cat_anp]
accum_cat = [i+'_acc' for i in cat_cols]


# %%



# %%



# %%


# %% [markdown]
# ## create list of flg_valid at certain date for each policy of each buyer_id

# %%
### create list_flg_valid to check if each policy was still valid at given issue_date

df['list_flg_valid'] = df.apply(lambda row: [bool(max([row['list_flg_inforce'][i], float(row['list_st_date'][i]>=row['issue_date'])])) if i<row['policy_seq'] else False for i,v in enumerate(row['list_st_date'])],axis=1)
df['insured_amt_acc'] = df.apply(lambda row: [sum(x) for x in zip(*np.array(row['list_insured_amt'])[np.array(row['list_flg_valid'])][:row['policy_seq']])],axis=1)
df['anp_acc'] = df.apply(lambda row: [sum(x) for x in zip(*np.array(row['list_anp'])[np.array(row['list_flg_valid'])][:row['policy_seq']])],axis=1)

### create accummulated insured_amt and anp by cat

for i,col in enumerate(accum_insured):
    df[col] = df['insured_amt_acc'].apply(lambda x: x[i])

for i,col in enumerate(accum_anp):
    df[col] = df['anp_acc'].apply(lambda x: x[i])

for i,col in enumerate(accum_cat):
    df[col] = df['insured_amt_acc'].apply(lambda x: float(x[i]>0))

# %% [markdown]
# ## calculate prod rec from index (segment's pen rate/ enterprise's pen rate)

# %%
### calculate prod rec from index

segment_cols = ['owner_age_grp', 'owner_gender', 'region_dummy']
penetration_cols = ['health_acc', 'investment_acc', 'protection_acc', 
                    'retirement_acc', 'savings_acc', 'ul_acc', 'nul_acc', 
                    'self_acc', 'dependent_acc']
insured_amt_cols = ['health_insured_amt_acc', 'investment_insured_amt_acc',
                    'protection_insured_amt_acc', 'retirement_insured_amt_acc',
                    'savings_insured_amt_acc', 'ul_insured_amt_acc',
                    'nul_insured_amt_acc', 'self_insured_amt_acc',
                    'dependent_insured_amt_acc']

enterprise_penet_dict = df[penetration_cols].mean().to_dict()
df_segment = df.groupby(by=segment_cols,as_index=False)[penetration_cols].mean()

for col,val in enterprise_penet_dict.items():
    enterprise_col = str(col).replace('_acc','_etp')
    df_segment[enterprise_col] = val

for col in enterprise_penet_dict.keys():
    index_col = str(col).replace('_acc','_idx')
    enterprise_col = str(col).replace('_acc','_etp')
    df_segment[index_col] = df_segment.apply(lambda row: round(row[col]/row[enterprise_col],2),axis=1)
    renamed_col = str(col).replace('_acc','_pen')
    df_segment = df_segment.rename(columns={col:renamed_col})

idx_col_to_exclude = ['self_idx','nul_idx']
idx_col_list = [i for i in df_segment.columns.values if ('_idx' in i) and (i not in idx_col_to_exclude)]

df_segment['prod_rec'] = df_segment[idx_col_list].idxmax(axis=1).apply(lambda x: x.replace('_idx',''))

### calculate median insured amt for each category

median_insured_amt_dict = {}
for col in insured_amt_cols:
    df_temp = df.iloc[np.where(df[col]>0)].groupby(by=segment_cols)[col].median().apply(round).to_frame()
    df_temp.columns = [str(col).replace('_insured_amt_acc','')]
    temp_dict = df_temp.to_dict('index')
    for key,val in temp_dict.items():
        if key in median_insured_amt_dict.keys():
            for nested_key,nested_val in val.items():
                if nested_key not in median_insured_amt_dict[key]:
                    median_insured_amt_dict[key][nested_key] = nested_val
        else: #key not in median_insured_amt_dict
            median_insured_amt_dict[key] = val

df_segment['key'] = df_segment.apply(lambda row: (row['owner_age_grp'],row['owner_gender'],row['region_dummy']),axis=1)
df_segment['prod_rec_median_insured_amt'] = df_segment.apply(lambda row: median_insured_amt_dict[row['key']][row['prod_rec']],axis=1)

prod_rec_dict = df_segment[['key','prod_rec','prod_rec_median_insured_amt']].set_index('key').to_dict('index')

# %% [markdown]
# ## Save recommendation for each cohort

# %%
## save prepocessor

pickle.dump(prod_rec_dict, open('./trg/prod_rec_dict.pkl', 'wb'))


