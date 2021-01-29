# flake8: noqa: E501

import numpy as np
import pandas as pd
import pmdarima
import datetime

class TSFM(object):
    def __init__(self,
                 train: pd.core.frame.DataFrame,
                 target_variable: str,
                 n_pred_period: int,
                 date_variable: [int, str] = 0,
                 value_variable: [int, str] = -1,
                 section_list: list = None,
                 cycle_length: int = 1,):
        self.target_variable = target_variable
        self.n_pred_period = n_pred_period
        # train and test df must have date as index, and 2 columns: sections(e.g. territory) and values(e.g. order_volume)
        if type(date_variable) is int:
            date_variable = train.columns[date_variable]
        if type(value_variable) is int:
            value_variable = train.columns[value_variable]
        # Select relevant columns for train and test df, create empty pred df
        self.columns = [date_variable, target_variable, value_variable]
        print(train)
        self.train = train[self.columns].groupby(
            self.columns[0:2], as_index=False).sum()
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
            temp_train_df = self.train.query(
                self.columns[1] + "==" + "'" + section + "'")[[self.columns[0], self.columns[2]]]
            temp_train_df.set_index(date_variable, inplace=True)
            if temp_train_df.shape[0] >= 2 * cycle_length:
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
                pred = arima_model.predict(self.n_pred_period)
                # date_generator = DateGenerator(start_date=max(self.train[self.columns[0]]))
                temp_pred_df = pd.DataFrame(
                    data={
                        self.columns[0]: pd.date_range(max(temp_train_df[self.columns[0]]),freq='MS',periods=self.n_pred_period+1)[1:],
                        self.columns[1]: [section for x in range(len(pred))],
                        self.columns[-1]: pred})  # Use numbers inplace of future dates for now)
                self.pred = self.pred.append(temp_pred_df, ignore_index=True)
                temp_pred_df = temp_pred_df[[self.columns[0], self.columns[2]]]
                temp_pred_df.set_index(date_variable, inplace=True)
                self.df_dict[section] = [temp_train_df, temp_pred_df]
                self.model_dict[section] = arima_model
            else:
                print("Number of data points in Section", section, "is too small (" + str(
                    temp_train_df.shape[0]) + ". Must be at least twice the declared cycle length.")


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

