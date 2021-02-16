# flake8: noqa: E501

import numpy as np
import pandas as pd
import pmdarima
from datetime import datetime
import typing
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

## Changes to be made
## 1 would like to add an argument 

class TSFM(object):
    def __init__(self,
                 df: pd.core.frame.DataFrame,
                 n_pred_period: int,
                 date_variable: typing.Union[int, str],
                 value_variable: typing.Union[int, str],
                 target_variable: str,
                 stop_date: str,         # stop date of train set, to split df to train and test sets
                 section_list: list = None,
                 cycle_length: int = 12,
                 alpha: float = 0.05,):
        self.target_variable = target_variable
        # self.n_pred_period = n_pred_period + abs((datetime.strptime(df[date_variable].to_numpy()[-1])  - datetime.strptime(stop_date, "%Y-%m-%d")).days)
        self.n_pred_period = n_pred_period
        self.stop_date = stop_date
        self.cycle_length = cycle_length
        # train and test df must have date as index, and 2 columns: sections(e.g. territory) and values(e.g. order_volume)
        if type(date_variable) is int:
            date_variable = df.columns[date_variable]
        if type(value_variable) is int:
            value_variable = df.columns[value_variable]
        # Select relevant columns for train and test df, create empty pred df
        self.columns = [date_variable, target_variable, value_variable]
        df[self.columns[0]] = pd.to_datetime(df[self.columns[0]])
        

        self.df = df
        self.pred = pd.DataFrame(columns=self.columns)

        # keys: sections(territories), value: list(train, test, pred), for easy storing and fetching data
        self.df_dict = dict()
        self.model_dict = dict()
        self.adjusted_model_dict = dict()

        # Iterate through the unique sections
        self.section_list = section_list
        if self.section_list is None:
            self.section_list = df[target_variable].unique()
        for section in self.section_list:
            print("Inspecting", section, "...")
            temp_actual_df = self.get_actual_data(section=section, is_adjusted=False)
            if temp_actual_df.shape[0] >= 2 * cycle_length:
                # train actual data
                print("Training", temp_actual_df.shape[0], "actual records ...")
                arima_model = pmdarima.auto_arima(temp_actual_df[temp_actual_df.columns[-1]],
                                                start_p=0, start_P=0,
                                                start_q=0, start_Q=0,
                                                d=1, D=1,
                                                max_p=4, max_P=2,
                                                max_d=2, max_D=2,
                                                max_q=2, max_Q=2,
                                                trace=True, m=cycle_length)
                self.model_dict[section] = arima_model

                temp_adjusted_actual_df = self.get_actual_data(section=section, is_adjusted=True)
                # train adjusted actual data
                print("Training", temp_adjusted_actual_df.shape[0], "adjusted actual records ...")
                arima_model = pmdarima.auto_arima(temp_adjusted_actual_df[temp_adjusted_actual_df.columns[-1]],
                                                start_p=0, start_P=0,
                                                start_q=0, start_Q=0,
                                                d=1, D=1,
                                                max_p=4, max_P=2,
                                                max_d=2, max_D=2,
                                                max_q=2, max_Q=2,
                                                trace=True, m=cycle_length)
                self.adjusted_model_dict[section] = arima_model
            else:
                print("Number of data points in Section", section, "is too small (" + str(
                    temp_actual_df.shape[0]) + ". Must be at least twice the declared cycle length.")

    # DF Getters--------------------------------------------------------------
    def get_actual_data(self, section: str, is_adjusted: bool = True,):
        agg_df = self.df[self.columns].groupby(self.columns[0:2], as_index=False).sum()
        agg_df = agg_df.query(self.columns[1] + "==" + "'" + section + "'")[[self.columns[0], self.columns[2]]]
        agg_df.set_index(self.columns[0], inplace=True)
        agg_df = TSFM.to_monthly(agg_df)
        if is_adjusted:
            return self.anomaly_filter(agg_df)
        return agg_df

    def get_train_data(self, section: str,):  ## added stop date
        actual_df = self.get_actual_data(section)
        return actual_df.iloc[lambda x: x.index <= self.stop_date]

    def get_test_data(self, section: str,):
        actual_df = self.get_actual_data(section)
        return actual_df.iloc[lambda x: x.index > self.stop_date]

    def get_pred_data(self, section: str, return_conf_int: bool = False, is_adjusted: bool = True):
        actual_df = self.get_actual_data(section)
        model = self.get_model(section=section, is_adjusted=is_adjusted)
        print(model.predict(self.n_pred_period, return_conf_int=return_conf_int))
        pred, conf_int = model.predict(self.n_pred_period, return_conf_int=True)
            
        # date_generator = DateGenerator(start_date=max(self.train[self.columns[0]]))
        temp_pred_df = pd.DataFrame(
            data={
                self.columns[0]: pd.date_range(max(actual_df.index),freq='MS',periods=self.n_pred_period+1)[1:],
                self.columns[1]: [section for x in range(len(pred))],
                self.columns[-1]: pred})  # Use numbers inplace of future dates for now)
        self.pred = self.pred.append(temp_pred_df, ignore_index=True)
        temp_pred_df = temp_pred_df[[self.columns[0], self.columns[2]]]
        temp_pred_df.set_index(self.columns[0], inplace=True)
        if return_conf_int:
            return temp_pred_df, conf_int
        return temp_pred_df

    # Model Getters----------------------------------------------------------
    def get_model(self, section: str, is_adjusted: bool = True):
        if is_adjusted:
            return self.adjusted_model_dict[section]
        return self.model_dict[section]
    
    # Plot Function-----------------------------------------------------------
    def plot(self, section: str):
        actual = self.get_actual_data(section, is_adjusted=False)
        adjusted_actual, conf_int_df = self.anomaly_filter(actual, return_conf_int=True)
        # pred, ci = self.get_pred_data(section, return_conf_int=True)

        actual_pred = self.get_pred_data(section, is_adjusted=False)
        adjusted_actual_pred = self.get_pred_data(section, is_adjusted=True)

        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(actual.index, actual[actual.columns[0]],label="Actual")   #Actuals This should come from original DS (all actuals)
        ax.plot(adjusted_actual.index, adjusted_actual[adjusted_actual.columns[0]],'-g', label="Adjusted Actual")   
        # ax.plot(pred.index, pred[pred.columns[0]], '-r',alpha=0.75,label="Forecast")  ## Pred
        ax.fill_between(conf_int_df.index, conf_int_df.iloc[:, 0], conf_int_df.iloc[:, 1],alpha=0.3, color='b')  ## Conf intervals
        
        ax.plot(actual_pred.index, actual_pred.iloc[:, 0], '--b',alpha=0.75,label="Actual Forecast")
        ax.plot(adjusted_actual_pred.index, adjusted_actual_pred.iloc[:, 0], '--g',alpha=0.75,label="Adjusted Actual Forecast")
        plt.title('Forecast Model')
        plt.xlabel('Year')
        plt.ylabel('Forecast Accurary')

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())

        plt.legend()
        plt.show()

    def anomaly_filter(self,
                       df: pd.core.frame.DataFrame,
                       return_conf_int: bool = False, 
                       n_rolling_period: int = 12,
                       alpha: float = 0.05):
        train = df.iloc[lambda x: x.index <= self.stop_date]
        stop_date = max(train.index)
        returning_ic_list = list()
        for i in range(train.shape[0], df.shape[0], 12):
            arima_model = pmdarima.auto_arima(train[train.columns[0]],
                                                start_p=0, start_P=0,
                                                start_q=0, start_Q=0,
                                                d=1, D=1,
                                                max_p=4, max_P=2,
                                                max_d=2, max_D=2,
                                                max_q=2, max_Q=2,
                                                trace=True, m=self.cycle_length)
            temp_actual_df = df.iloc[i:min(i+12, df.shape[0]), :]
            temp_pred, ic_list = arima_model.predict(n_rolling_period, return_conf_int=True, alpha=alpha)
            for j in range(temp_actual_df.shape[0]):
                temp_actual = temp_actual_df.iloc[j, 0]
                ic = ic_list[j]
                if temp_actual < ic[0] or temp_actual > ic[1]:
                    temp_actual_df.iloc[j, 0] = temp_pred[j]
            train = train.append(temp_actual_df)
            returning_ic_list = returning_ic_list + ic_list.tolist()
        if return_conf_int:
            returning_ic_list = np.array(returning_ic_list)
            ic_df = pd.DataFrame(
                data={
                    'lower': returning_ic_list[:, 0],
                    'upper': returning_ic_list[:, 1],
                },
                index=pd.date_range(stop_date,freq='MS',periods=returning_ic_list.shape[0]+1)[1:]
            )
            return train, ic_df
        return train


    @classmethod
    def to_monthly(cls, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return df.resample('MS').sum()
