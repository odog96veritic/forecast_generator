# TSFM
TSFM (Time-Series Forecasting Machine) automates time series forecasting and provides anomaly detection solutions given the monthly data as input.
```python
class TSFM (
    self,
    df: pd.core.frame.DataFrame,
    n_pred_period: int,
    date_variable: typing.Union[int, str],
    target_variable: typing.Union[int, str],
    value_variable: typing.Union[int, str],
    stop_date: str,
    section_list: list = None,
    cycle_length: int = 12,
    alpha: float = 0.05,
    stepwise: bool = True,
    start_order: tuple = (0, 1, 0),
    max_order: tuple = (4, 2, 5),
    start_seasonal_order: tuple = (0, 1, 0),
    max_seasonal_order: tuple = (2, 2, 4))
```
***Arguments***
* **df**: *pd.core.frame.DataFrame*: The input dataframe, can be monthly or daily. May comprise of data from different items (sections).
* **n_pred_period**: *int*: The number of predictions to generate from the input data for each item (section).
* **date_variable**: *typing.Union[int, str]*: Indicates the date column of the input dataframe. Can be string name or index integer.
* **target_variable**: *typing.Union[int, str]*: Indicates the item(section) column, i.e. product_name, of the input dataframe. Can be string name or index integer.
* **value_variable**: *typing.Union[int, str]*: Indicates the value data column, i.e. sales, of the input dataframe. Can be string name or index integer.
* **stop_date**: *str*: Last date of the "gold-standard", historical time segment. All data before or on this date are not adjusted by the anomaly filter.
* **section_list**: *list*: List of items (sections) to be extracted from the input dataframe and processed. Default of **None** means all sections are processed.
* **cycle_length**: *int*: Length, in month, of one cycle. Default of **12**.
* **alpha**: *float*: Alpha value to determine the confidence interval in anomaly filter. Default of **0.05**.
* **stepwise**: *bool*: If the stepwise algorithm is used in auto_arima procedure. Default of **True**.
* **start_order**: *tuple*: The starting (p, d, q) values used in auto_arima procedure. Default of **(0, 1, 0)**.
* **max_order**: *tuple*: The max (p, d, q) values used in auto_arima procedure. Default of **(4, 2, 5)**.
* **start_seasonal_order**: *tuple*: The starting (P, D, Q) values used in auto_arima procedure. Default of **(0, 1, 0)**.
* **max_seasonal_order**: *tuple*: The max (P, D, Q) values used in auto_arima procedure. Default of **(2, 2, 4)**.

***Attributes***
* Includes all names mentioned in the argument section above.
* **is_log_transformed_dict**: Dictionary. Keys are string names of an item (section), values are boolean values signal if data of that section is log transformed.
* **pred_dict**: Dictionary. Keys are string names of an item (section), values are prediction data of that section.
* **pred_ic_dict**: Dictionary. Keys are string names of an item (section), values are the confidence intervals, each in form of a list of two, [lower_bound, upper_bound].
* **adjusted_pred_dict**: Dictionary. Keys are string names of an item (section), values are prediction data of that section after being adjusted by the anomaly filter.
* **adjusted_pred_ic_dict** : Dictionary. Keys are string names of an item (section), values are the confidence intervals after being adjusted by the anomaly filter. Each value is in form of a list of two, [lower_bound, upper_bound].

***Methods***
* **get_actual_data(self, section: str, is_adjusted: bool, is_log_transformed: bool = None)**: Returns the input data of a section with or without transformations. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
    * **is_adjusted**: *bool*: determine if the returning data is adjusted by the anomally filter.
    * **is_log_transformed**: *bool*: determine if the returning data is log transformed. Default of **None** menas it's value is determined by self.is_log_transformed_dict[section].
* **get_train_data(self, section: str)**: Returns the training data for the anomaly filter method. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
* **get_test_data(self, section: str)**: Returns the testing data for the anomaly filter method, a.k.a validating segment of the historical data. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
* **get_pred_data(self, section: str, is_adjusted: bool, return_conf_int: bool = False)**: Returns predictions a section either modeled from trainsformed or untransformed training data. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
    * **is_adjusted**: *bool*: determines if the returning data are modeled from the adjusted training data.
    * **return_conf_int**: *bool*: determines if the confidence intervals are returned with the prediction data. Default of False means ICs are not returned.
* **get_pred_df(self)**: Returns formated df that can be written to DB in Django backend.
* **get_model(self, section: str, is_adjusted: bool)**: Returns a trained model of a section. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
    * **is_adjusted**: *bool*: determine if the model is trained from the adjusted or unadjusted data.
* **plot(self, section: str)**: Plot function used for development. Plotting both actual and adjusted historical data and predictions. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
* **anomaly_filter(self, df: pd.core.frame.DataFrame, return_conf_int: bool = False, n_rolling_period: int = 12, alpha: float = 0.05)**: Identifies anomallies and replace them with predictions from models trained from adjusted data ranging from the beginning to **self.stop_date**. This procedure is based on the time-series cross-validation process (train and eval are selected, returning the QC metrics, add eval to train then select new eval). Instead of returning QC metrics:
    * arima model is trained
    * data in eval is adjusted
    * new eval is added to train
    * new eval is selected
    * repeat until all historical data are consumed.
    
    Parameters:
    * **df**: *pd.core.frame.DataFrame*: Input dataframe of one item (section). Df should be monthly, have date indeces, 1 value column.
    * **return_conf_int**: *bool*: determines if the confidence interval dataframe is returned with the adjusted data. Default of False means IC df is not returned.
    * **n_rolling_period**: *int*: Size of eval segment. Default of 12.
    * **alpha**: *float*: Alpha value to determine the confidence interval. Default of **0.05**.
* **cross_validate(self,section: str, is_adjusted: bool)**: Performs time-series cross validation and print QC metrics. Parameters:
    * **section**: *str*: an element in the unique list of target variables.
    * **is_adjusted**: *bool*: determine if the validation is from the adjusted or unadjusted data.