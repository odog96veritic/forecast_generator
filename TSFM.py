# flake8: noqa: E501

import numpy as np
import pandas as pd
import pmdarima
import datetime
import typing
from matplotlib import pyplot as plt

class TSFM(object):
    def __init__(self,
                 train: pd.core.frame.DataFrame,
                 target_variable: str,
                 n_pred_period: int,
                 date_variable: typing.Union[int, str] = 0,
                 value_variable: typing.Union[int, str] = -1,
                 section_list: list = None,
                 cycle_length: int = 12,
                 do_anomaly_filter : bool = True,
                 alpha: float = 0.05,):
        self.target_variable = target_variable
        self.n_pred_period = n_pred_period
        self.cycle_length = cycle_length
        # train and test df must have date as index, and 2 columns: sections(e.g. territory) and values(e.g. order_volume)
        if type(date_variable) is int:
            date_variable = train.columns[date_variable]
        if type(value_variable) is int:
            value_variable = train.columns[value_variable]
        # Select relevant columns for train and test df, create empty pred df
        self.columns = [date_variable, target_variable, value_variable]

        train[self.columns[0]] = pd.to_datetime(train[self.columns[0]])

        self.train = train
        self.pred = pd.DataFrame(columns=self.columns)

        # keys: sections(territories), value: list(train, test, pred), for easy storing and fetching data
        self.df_dict = dict()
        self.model_dict = dict()

        # Iterate through the unique sections
        self.section_list = section_list
        if self.section_list is None:
            self.section_list = train[target_variable].unique()
        for section in self.section_list:
            print("Inspecting", section, "...")
            # Query data for one selected section
            temp_train_df = self.get_train_data(section=section)
            if temp_train_df.shape[0] >= 2 * cycle_length:
                if do_anomaly_filter:
                    temp_train_df = self.anomaly_filter(temp_train_df, alpha=alpha)
                print("Training", temp_train_df.shape[0], "records ...")
                # set Date as Index
                # temp df now has 2 columns: section(territory) and value(order_volume)
                arima_model = pmdarima.auto_arima(temp_train_df[temp_train_df.columns[-1]],
                                                start_p=0, start_P=0,
                                                start_q=0, start_Q=0,
                                                d=1, D=1,
                                                max_p=4, max_P=2,
                                                max_d=2, max_D=2,
                                                max_q=2, max_Q=2,
                                                trace=True, m=cycle_length)
                self.model_dict[section] = arima_model
            else:
                print("Number of data points in Section", section, "is too small (" + str(
                    temp_train_df.shape[0]) + ". Must be at least twice the declared cycle length.")

    def get_train_data(self, section: str,):
        agg_df = self.train[self.columns].groupby(self.columns[0:2], as_index=False).sum()
        agg_df = agg_df.query(self.columns[1] + "==" + "'" + section + "'")[[self.columns[0], self.columns[2]]]
        agg_df.set_index(self.columns[0], inplace=True)
        agg_df = TSFM.to_monthly(agg_df)
        return agg_df

    def get_pred_data(self, section: str, train_end_date):
        train_df = self.get_train_data(section)
        model = self.model_dict[section]
        pred = model.predict(self.n_pred_period)
        # date_generator = DateGenerator(start_date=max(self.train[self.columns[0]]))
        temp_pred_df = pd.DataFrame(
            data={
                self.columns[0]: pd.date_range(max(train_df.index),freq='MS',periods=self.n_pred_period+1)[1:],
                self.columns[1]: [section for x in range(len(pred))],
                self.columns[-1]: pred})  # Use numbers inplace of future dates for now)
        self.pred = self.pred.append(temp_pred_df, ignore_index=True)
        temp_pred_df = temp_pred_df[[self.columns[0], self.columns[2]]]
        temp_pred_df.set_index(self.columns[0], inplace=True)
        return 


    def plot(self, section: str):
        [train_df, pred_df] = self.df_dict[section]
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle("Prediction at " + self.columns[1] + " level (" + section +  ")")
        ax.set_xlabel("Periods")
        ax.set_ylabel(self.columns[-1])
        train_df.plot(ax=ax, marker='o')
        pred_df.plot(ax=ax, marker='o')
        ax.legend(["train", "prediction"])
        plt.show()

    def get_df(self):
        return_df = self.train
        for section in self.section_list:
            pred = pd.DataFrame.copy(self.df_dict[section][1])
            pred.reset_index(inplace=True)
            pred[self.target_variable] = [section for x in range(pred.shape[0])]
            return_df = return_df.append(pred, ignore_index=True)
        return_df.sort_values(by=[self.columns[1], self.columns[0]], inplace=True, ignore_index=True)
        return return_df

    def get_pred_df(self):
        return_df = pd.DataFrame(columns=self.columns)
        for section in self.section_list:
            pred = pd.DataFrame.copy(self.df_dict[section][1])
            pred.reset_index(inplace=True)
            pred[self.target_variable] = [section for x in range(pred.shape[0])]
            return_df = return_df.append(pred, ignore_index=True)
        return_df.sort_values(by=[self.columns[1], self.columns[0]], inplace=True, ignore_index=True)
        return return_df

    def is_anomalous(self, section: str, actual_value: float, date: datetime.datetime):
        model = self.model_dict[section]
        future_date_index = self.df_dict[section][0].index[-1].month - date.month
        print("current_date = {}, pred_date = {}".format(self.df_dict[section][0].index[-1], date))
        print([self.df_dict[section][0].shape[0] + future_date_index - 1])
        pred = model.fit_predict([self.df_dict[section][0].shape[0] + future_date_index - 1])[-1]
        alpha = 0.05
        conf_int = model.conf_int(alpha = alpha)
        extreme_conf_int = model.conf_int(alpha = 0.0005)

        # zscore = (actual_value - forecast[0][-1]) / max(float(forecast[1][-1]), float_min)
        # anomaly_probability = (2 * st.norm(0, 1).cdf(abs(zscore))) - 1
        
        result = {'Prediction': float(pred) if not float(
                        pred) == float('inf') else 0.0,
                    'CILower': float(conf_int[0]) if not float(
                        conf_int[0]) == float('-inf') else 0.0,
                    'CIUpper': float(conf_int[1]) if not float(
                        conf_int[1]) == float('inf') else 0.0,
                    'ConfLevel': float(1.0 - alpha) * 100,
                    'IsAnomaly': actual_value < conf_int[0] or actual_value > conf_int[1],
                    'IsAnomalyExtreme': actual_value < extreme_conf_int[0] or actual_value > extreme_conf_int[1],}
                    # 'AnomalyProbability': 1 if raw_actual is None else float(anomaly_probability),
                    # 'DownAnomalyProbability': 1 if raw_actual is None else float(down_anomaly_probability),
                    # 'UpAnomalyProbability': 1 if raw_actual is None else float(up_anomaly_probability),
                    # 'ModelFreshness': model_freshness}
        print(result)
        return(result)
    
    def anomaly_filter(self, df: pd.core.frame.DataFrame, train_size = 24, alpha: float = 0.05):
        '''
        input df is monthly
        input df has date as index and 1 value column
        '''
        def find_anomaly_index(actual_list: list, conf_int_list: list, inf_bound: int = None):
            print("finding Anomaly Index")
            print(actual_list, conf_int_list)
            for i in range(len(actual_list)):
                actual = actual_list[i]
                conf_int =  conf_int_list[i]
                if inf_bound is not None:
                    conf_int[inf_bound] = np.inf
                if actual < conf_int[0] or actual > conf_int[1]:
                    return i
            return None

        train = df.head(train_size)
        inspecting_df = df.tail(df.shape[0] - train_size)
        
        while train.shape[0] < df.shape[0]:
            arima_model = pmdarima.auto_arima(train[train.columns[0]],
                                            start_p=0, start_P=0,
                                            start_q=0, start_Q=0,
                                            d=1, D=1,
                                            max_p=4, max_P=2,
                                            max_d=2, max_D=2,
                                            max_q=2, max_Q=2,
                                            trace=True, m=self.cycle_length)
            pred_list, conf_int_list = arima_model.predict(n_periods=df.shape[0] - train.shape[0], return_conf_int=True, alpha=alpha)
            anomaly_index = find_anomaly_index(df.iloc[train.shape[0]:][df.columns[0]], conf_int_list)
            if anomaly_index is not None:
                print("Anomaly spotted at {}".format(train.shape[0] + anomaly_index))
                if anomaly_index != 0:
                    train = train.append(df.iloc[train.shape[0]:train.shape[0] + anomaly_index])
                arima_model = pmdarima.auto_arima(train[train.columns[0]],
                                                    start_p=0, start_P=0,
                                                    start_q=0, start_Q=0,
                                                    d=1, D=1,
                                                    max_p=4, max_P=2,
                                                    max_d=2, max_D=2,
                                                    max_q=2, max_Q=2,
                                                    trace=True, m=self.cycle_length)
                pred_list, conf_int_list = arima_model.predict(n_periods=1, return_conf_int=True, alpha=alpha)
                pred, conf_int = pred_list[0], conf_int_list[0]
                actual = df.iloc[train.shape[0]][df.columns[0]]
                print("Pred {}, actual {}, conf_int {}".format(pred, actual, conf_int))
                if actual < conf_int[0] or actual > conf_int[1]:
                    if actual > conf_int[1]:
                        print("Appending actual data")
                        train = train.append(df.iloc[train.shape[0]])
                    else:
                        print("Appending modeled data")
                        temp_pred_df = pd.DataFrame(
                            data={train.columns[0]: [pred]}, index=pd.date_range(max(train.index),freq='MS',periods=2)[1:])
                        train = train.append(temp_pred_df)
            else:
                print("No Anomaly spotted, appending the rest of the data")
                train = train.append(df.iloc[train.shape[0]:])
        return train

    def __column_type_validate(self, column: str, dtypes: list):
        if self.train.dtypes[column] in dtypes:
            return
        raise TypeError("Invalid dtype for column", column, ".")

    @classmethod
    def to_monthly(cls, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return df.resample('MS').sum()

    
    
