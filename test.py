import pandas as pd
import numpy as np
import datetime
from TSFM import TSFM

df = pd.read_csv('ord_vol.csv')
df['beg_month'] = pd.to_datetime(df['beg_month'])

tsfm = TSFM(df=df, n_pred_period=24, date_variable='beg_month', target_variable='product', value_variable='ro', stop_date="2020-03-01", section_list=["HOME"])
tsfm.plot("HOME")