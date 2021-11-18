import pandas as pd
import numpy as np
import datetime


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def calculate_anp(premium,pay_method):
    if pd.isnull(premium) or (str(premium).replace('.','').replace(',','').isdigit()==False):
        anp = 0.0
    elif pay_method == 'Single Pay':
        anp = float(str(premium).replace(',',''))/10.0
    else:
        anp = float(str(premium).replace(',',''))
    return anp

def age_grp(age):
    if pd.isnull(age):
        age_grp = np.nan
    elif age <20:
        age_grp = 'a.<20'
    elif age <25:
        age_grp = 'b.20-24'
    elif age<30:
        age_grp = 'c.25-29'
    elif age <35:
        age_grp = 'd.30-34'
    elif age<40:
        age_grp = 'e.35-39'
    elif age <45:
        age_grp = 'f.40-44'
    elif age<50:
        age_grp = 'g.45-49'
    elif age <55:
        age_grp = 'h.50-54'
    elif age<60:
        age_grp = 'i.55-59'
    else:
        age_grp = 'j.60up'
    return age_grp

def occupation_grp(occ):
    if str(occ).strip() == 'STUDENT':
        retval = 'a.student'
    elif str(occ).strip() in ['HOUSEWIFE','UNEMPLOYED']:
        retval = 'b.non-salaried'
    elif str(occ).strip() == 'PENSIONER':
        retval = 'c.pensioner'
    elif str(occ).strip() == 'OVERSEAS CONTRACT WORKER (OCW)':
        retval = 'd.ocw'
    elif str(occ).strip() == 'OFFICE WORKER':
        retval = 'e.office worker'
    elif str(occ).strip() == 'BUSINESSMAN/ / BUSINESSWOMAN':
        retval = 'g.businessman/woman'
    elif pd.isnull(occ):
        retval = None
    else:
        retval = 'f.professionals'
    return retval



def pol_st_grp(pol_st):
    if str(pol_st).strip() in ['inforce','claimed','cancelled']:
        pol_st_grp = str(pol_st).strip()
    elif 'lapse' in pol_st:
        pol_st_grp = 'lapsed'
    elif 'surrender' in pol_st:
        pol_st_grp = 'surrendered'
    else:
        pol_st_grp = 'others'
    return pol_st_grp