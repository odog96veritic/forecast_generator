# flake8: noqa: E501

import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from pandas.core.indexes.api import get_objs_combined_axis
import pmdarima
from datetime import datetime
import typing
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error # upgrade to the the latest scikit-learn pip install -U scikit-learn
import random

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
                 alpha: float = 0.05,
                 stepwise: bool = True,
                 start_order: tuple = (0, 1, 0),
                 max_order: tuple = (4, 2, 5),
                 start_seasonal_order: tuple = (0, 1, 0),
                 max_seasonal_order: tuple = (2, 2, 4)):
        self.target_variable = target_variable
        # self.n_pred_period = n_pred_period + abs((datetime.strptime(df[date_variable].to_numpy()[-1])  - datetime.strptime(stop_date, "%Y-%m-%d")).days)
        self.n_pred_period = n_pred_period
        self.stop_date = stop_date
        self.cycle_length = cycle_length
        self.alpha = alpha
        self.stepwise = stepwise
        self.start_order = start_order
        self.max_order = max_order
        self.start_seasonal_order = start_seasonal_order
        self.max_seasonal_order = max_seasonal_order

        self.is_log_transformed_dict = dict()
        self.pred_dict = dict()
        self.pred_ic_dict = dict()
        self.adjusted_pred_dict = dict()
        self.adjusted_pred_ic_dict = dict()

        # train and test df must have date as index, and 2 columns: sections(e.g. territory) and values(e.g. order_volume)
        if type(date_variable) is int:
            date_variable = df.columns[date_variable]
        if type(value_variable) is int:
            value_variable = df.columns[value_variable]
        # Select relevant columns for train and test df, create empty pred df
        self.columns = [date_variable, target_variable, value_variable]
        df[self.columns[0]] = pd.to_datetime(df[self.columns[0]])
        

        self.df = df.copy()

        # keys: sections(territories), value: list(train, test, pred), for easy storing and fetching data
        self.df_dict = dict()
        self.model_dict = dict()
        self.adjusted_model_dict = dict()

        # Iterate through the unique sections
        self.section_list = section_list
        if self.section_list is None:
            self.section_list = df[target_variable].unique()
        for section in self.section_list:
            self.is_log_transformed_dict[section] = False
            models = self.__train_models(section = section, is_log_transformed=False)
            if self.__have_negative_prediction(models=models):
                print("Negative prediction detected in", section)
                self.is_log_transformed_dict[section] = True
                models = self.__train_models(section = section, is_log_transformed=True)
            (arima_model, adjusted_arima_model) = models
            self.model_dict[section] = arima_model
            self.adjusted_model_dict[section] = adjusted_arima_model
            self.pred_dict[section], self.pred_ic_dict[section] = self.__get_pred_data(section=section, return_conf_int=True, is_adjusted=False)
            self.adjusted_pred_dict[section], self.adjusted_pred_ic_dict[section] = self.__get_pred_data(section=section, return_conf_int=True, is_adjusted=True)

            # print("Inspecting", section, "...")
            # temp_actual_df = self.get_actual_data(section=section, is_adjusted=False)
            # if temp_actual_df.shape[0] >= 2 * cycle_length:
            #     # train actual data
            #     print("Training", temp_actual_df.shape[0], "actual records ...")
            #     arima_model = pmdarima.auto_arima(temp_actual_df[temp_actual_df.columns[-1]],
            #                                     start_p=start_p, start_P=start_P,
            #                                     start_q=start_q, start_Q=start_Q,
            #                                     d=start_d, D=start_D,
            #                                     max_p=max_p, max_P=max_P,
            #                                     max_d=max_d, max_D=max_D,
            #                                     max_q=max_q, max_Q=max_Q,
            #                                     trace=True, m=cycle_length, stepwise=stepwise)
            #     self.model_dict[section] = arima_model

            #     temp_adjusted_actual_df = self.get_actual_data(section=section, is_adjusted=True)
            #     # train adjusted actual data
            #     print("Training", temp_adjusted_actual_df.shape[0], "adjusted actual records ...")
            #     arima_model = pmdarima.auto_arima(temp_adjusted_actual_df[temp_adjusted_actual_df.columns[-1]],
            #                                     start_p=start_p, start_P=start_P,
            #                                     start_q=start_q, start_Q=start_Q,
            #                                     d=start_d, D=start_D,
            #                                     max_p=max_p, max_P=max_P,
            #                                     max_d=max_d, max_D=max_D,
            #                                     max_q=max_q, max_Q=max_Q,
            #                                     trace=True, m=cycle_length, stepwise=stepwise)
            #     self.adjusted_model_dict[section] = arima_model
            # else:
            #     print("Number of data points in Section", section, "is too small (" + str(
            #         temp_actual_df.shape[0]) + ". Must be at least twice the declared cycle length.")

    # DF Getters--------------------------------------------------------------
    def get_actual_data(self, section: str, is_adjusted: bool, is_log_transformed: bool = False) -> pd.core.frame.DataFrame:
        agg_df = self.df[self.columns].groupby(self.columns[0:2], as_index=False).sum().copy()
        agg_df = agg_df.query(self.columns[1] + "==" + "'" + section + "'")[[self.columns[0], self.columns[2]]]
        agg_df.set_index(self.columns[0], inplace=True)
        agg_df = TSFM.to_monthly(agg_df)
        if is_adjusted:
            agg_df = self.anomaly_filter(agg_df, alpha = self.alpha)
        if is_log_transformed:
            agg_df = TSFM.log_transform(agg_df)
        return agg_df

    def get_train_data(self, section: str):  ## added stop date
        actual_df = self.get_actual_data(section, is_adjusted=False)
        return actual_df.iloc[lambda x: x.index <= self.stop_date].copy()

    def get_test_data(self, section: str, ):
        actual_df = self.get_actual_data(section)
        return actual_df.iloc[lambda x: x.index > self.stop_date].copy()

    def get_pred_data(self, section: str, return_conf_int: bool = False, is_adjusted: bool = True):
        pred, ic = None, None
        if is_adjusted:
            pred, ic = self.adjusted_pred_dict[section], self.adjusted_pred_ic_dict[section]
        else:
            pred, ic = self.pred_dict[section], self.pred_ic_dict[section]
        if return_conf_int:
            return pred, ic
        return pred

    def __get_pred_data(self, section: str, return_conf_int: bool = False, is_adjusted: bool = True):
        actual_df = self.get_actual_data(section, False)
        model = self.get_model(section=section, is_adjusted=is_adjusted)
        if model is None:
            print("Model of", section, "section was not initiated. Might due to insufficient training data.")
            return None
        print(model.predict(self.n_pred_period, return_conf_int=return_conf_int))
        pred, conf_int = model.predict(self.n_pred_period, return_conf_int=True)

        temp_pred_df = pd.DataFrame(
            data={
                self.columns[0]: pd.date_range(max(actual_df.index),freq='MS',periods=self.n_pred_period+1)[1:],
                self.columns[1]: [section for x in range(len(pred))],
                self.columns[-1]: pred})  # Use numbers inplace of future dates for now)
        temp_pred_df = temp_pred_df[[self.columns[0], self.columns[2]]]
        temp_pred_df.set_index(self.columns[0], inplace=True)
        if self.is_log_transformed_dict[section]:
            temp_pred_df = self.exp_transform(temp_pred_df)
        if return_conf_int:
            return temp_pred_df.copy(), conf_int
        return temp_pred_df.copy()

    def get_pred_df(self):
        '''
        function to be called in the backend to get pred data from all sections
        '''
        return_df = pd.DataFrame(columns=self.columns)
        for section in self.section_list:
            actual_pred = self.get_pred_data(section, is_adjusted=False)
            adjusted_actual_pred = self.get_pred_data(section, is_adjusted=True)
            pred = pd.DataFrame(data={actual_pred.columns[0]: actual_pred[actual_pred.columns[0]].to_numpy(), "value2": adjusted_actual_pred[adjusted_actual_pred.columns[0]].to_numpy()},
                                index = actual_pred.index)
            pred.reset_index(inplace=True)
            pred[self.target_variable] = [section for x in range(pred.shape[0])]
            return_df = return_df.append(pred, ignore_index=True)
        return_df.sort_values(by=[self.columns[1], self.columns[0]], inplace=True, ignore_index=True)
        return return_df

    # Model Getters----------------------------------------------------------
    def get_model(self, section: str, is_adjusted: bool = True):
        if is_adjusted:
            return self.adjusted_model_dict[section]
        return self.model_dict[section]
    
    # Plot Function-----------------------------------------------------------
    def plot(self, section: str):
        actual = self.get_actual_data(section, is_adjusted=False)
        adjusted_actual, conf_int_df = self.anomaly_filter(actual, return_conf_int=True, alpha = self.alpha)
        # pred, ci = self.get_pred_data(section, return_conf_int=True)

        actual_pred = self.get_pred_data(section, is_adjusted=False)
        adjusted_actual_pred = self.get_pred_data(section, is_adjusted=True)

        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(actual.index, actual[actual.columns[0]].to_numpy(),label="Actual")   #Actuals This should come from original DS (all actuals)
        ax.plot(adjusted_actual.index, adjusted_actual[adjusted_actual.columns[0]].to_numpy(),'-g', label="Adjusted Actual")   
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
        print("Applying anomaly filter...")
        df = df.copy()
        train = df.iloc[lambda x: x.index <= self.stop_date]
        stop_date = max(train.index)
        returning_ic_list = list()
        (start_p, start_d, start_q) = self.start_order
        (max_p, max_d, max_q) = self.max_order
        (start_P, start_D, start_Q) = self.start_seasonal_order
        (max_P, max_D, max_Q) = self.max_seasonal_order
        for i in range(train.shape[0], df.shape[0], n_rolling_period):
            arima_model = pmdarima.auto_arima(train[train.columns[0]],
                                                start_p=start_p, start_P=start_P,
                                                start_q=start_q, start_Q=start_Q,
                                                d=start_d, D=start_D,
                                                max_p=max_p, max_P=max_P,
                                                max_d=max_d, max_D=max_D,
                                                max_q=max_q, max_Q=max_Q,
                                                trace=True, m=self.cycle_length, stepwise=self.stepwise)
            temp_actual_df = df.iloc[i:min(i+n_rolling_period, df.shape[0]), :].copy()
            temp_pred, ic_list = arima_model.predict(n_rolling_period, return_conf_int=True, alpha=alpha)
            for j in range(temp_actual_df.shape[0]):
                temp_actual = temp_actual_df.iloc[j, 0]
                ic = ic_list[j]
                if temp_actual < ic[0] or temp_actual > ic[1]:
                    temp_actual_df.iloc[j, 0] = temp_pred[j]
            train = train.append(temp_actual_df)
            returning_ic_list = returning_ic_list + ic_list.tolist()
        train[train.columns[0]] = train[train.columns[0]].astype('float')
        train = train.asfreq('MS')
        if return_conf_int:
            returning_ic_list = np.array(returning_ic_list)
            ic_df = pd.DataFrame(
                data={
                    'lower': returning_ic_list[:, 0],
                    'upper': returning_ic_list[:, 1],
                },
                index=pd.date_range(stop_date,freq='MS',periods=returning_ic_list.shape[0]+1)[1:]
            )
            return train.copy(), ic_df
        return train.copy()
    
    def cross_validate(self,section: str, is_adjusted: bool):
        actual_df = self.get_actual_data(section, is_adjusted)
        actual_arr = actual_df[actual_df.columns[0]].to_numpy(dtype='float')
        trained_model = self.get_model(section, is_adjusted)
        order = trained_model.order
        seasonal_order = trained_model.seasonal_order

        model = pmdarima.ARIMA(order = order, seasonal_order=seasonal_order)
        MAE = list()
        MAPE = list()
        tscv = TimeSeriesSplit(10)
        for train_index, test_index in tscv.split(X=actual_df.index):
            if len(train_index) >= 24:
                print("train_index = {}".format(train_index))
                print("test_index = {}".format(test_index))
                train = actual_arr[train_index]
                test = actual_arr[test_index]
                print("train = {}".format(train))
                print("test = {}".format(test))
                model.fit(train)
                pred = model.predict(len(test))
                mae = mean_absolute_error(test, pred)
                mape = mean_absolute_percentage_error(test, pred)
                MAE.append(mae)
                MAPE.append(mape)
                
                print("MAE = {}".format(mae))
                print("MAPE = {}".format(mape))
                print("-"*50)
            else:
                pass
        print("MAE = {}".format(MAE))
        print("MAPE = {}".format(MAPE))
        print("Average MAE = {}".format(np.mean(MAE)))
        print("Average MAPE = {}".format(np.mean(MAPE)))

    def __train_models(self, section: str, is_log_transformed: bool) -> tuple:
        (start_p, start_d, start_q) = self.start_order
        (max_p, max_d, max_q) = self.max_order
        (start_P, start_D, start_Q) = self.start_seasonal_order
        (max_P, max_D, max_Q) = self.max_seasonal_order
    
        print("Inspecting", section, "...")
        temp_actual_df = self.get_actual_data(section=section, is_adjusted=False, is_log_transformed=is_log_transformed)
        if temp_actual_df.shape[0] >= 2 * self.cycle_length:
            # train actual data
            print("Training", temp_actual_df.shape[0], "actual records ...")
            arima_model = pmdarima.auto_arima(temp_actual_df[temp_actual_df.columns[-1]],
                                            start_p=start_p, start_P=start_P,
                                            start_q=start_q, start_Q=start_Q,
                                            d=start_d, D=start_D,
                                            max_p=max_p, max_P=max_P,
                                            max_d=max_d, max_D=max_D,
                                            max_q=max_q, max_Q=max_Q,
                                            trace=True, m=self.cycle_length, stepwise=self.stepwise)

            temp_adjusted_actual_df = self.get_actual_data(section=section, is_adjusted=True, is_log_transformed=is_log_transformed)
            # train adjusted actual data
            print("Training", temp_adjusted_actual_df.shape[0], "adjusted actual records ...")
            adjusted_arima_model = pmdarima.auto_arima(temp_adjusted_actual_df[temp_adjusted_actual_df.columns[-1]],
                                            start_p=start_p, start_P=start_P,
                                            start_q=start_q, start_Q=start_Q,
                                            d=start_d, D=start_D,
                                            max_p=max_p, max_P=max_P,
                                            max_d=max_d, max_D=max_D,
                                            max_q=max_q, max_Q=max_Q,
                                            trace=True, m=self.cycle_length, stepwise=self.stepwise)
            return arima_model, adjusted_arima_model
        print("Number of data points in Section", section, "is too small (" + str(
            temp_actual_df.shape[0]) + ". Must be at least twice the declared cycle length.")
        return None, None

    def __have_negative_prediction(self, models: tuple) -> bool:
        (model, adjusted_model) = models
        if model is None or adjusted_model is None:
            return False
        pred = model.predict(self.n_pred_period)
        adjusted_pred = adjusted_model.predict(self.n_pred_period)
        if min(min(pred), min(adjusted_pred)) < 0:
            return True
        return False

    @staticmethod
    def log_transform(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        df has 1 value column and has dates as indeces
        '''
        value_arr = df.loc[:, df.columns[0]].values.copy()
        value_arr[value_arr < 1] = 1 # avoid negative results from transformation
        value_arr = np.log(value_arr)
        return  pd.DataFrame(index=df.index, data={df.columns[0]: value_arr})
    
    @staticmethod
    def exp_transform(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        '''
        df has 1 value column and has dates as indeces
        '''
        value_arr = df.loc[:, df.columns[0]].values.copy()
        value_arr = np.exp(value_arr)
        return  pd.DataFrame(index=df.index, data={df.columns[0]: value_arr})

    @classmethod
    def to_monthly(cls, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return df.resample('MS').sum()

    def foo():
        return
