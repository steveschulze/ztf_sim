"""Sky brightness model."""

import sklearn
from sklearn import model_selection, ensemble, preprocessing, pipeline
from sklearn import neighbors, svm, linear_model
from sklearn_pandas import DataFrameMapper
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from .constants import FILTER_NAME_TO_ID, BASE_DIR


class SkyBrightness(object):
    """
    A class used to predict sky brightness based on various parameters.

    Attributes
    ----------
    clf_r : sklearn model
        A pre-trained model for predicting sky brightness in the 'r' filter.
    clf_g : sklearn model
        A pre-trained model for predicting sky brightness in the 'g' filter.
    clf_i : sklearn model
        A pre-trained model for predicting sky brightness in the 'i' filter.

    Methods
    -------
    __init__()
        Initializes the SkyBrightness class with pre-trained models.
    
    predict(df)
        Predicts sky brightness for the given dataframe with specific columns.
    """
    

    def __init__(self):
        """
        Initializes the SkyBrightness class by loading pre-trained models for
        'r', 'g', and 'i' filters from specified file paths.
        """
    
        """
        Predicts sky brightness for the given dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with columns:
            - mooonillf: float, 0-1
            - moonalt: float, degrees
            - moon_dist: float, degrees
            - azimuth: float, degrees
            - altitude: float, degrees
            - sunalt: float, degrees
            - filterkey: int, 1, 2, 3

        Returns
        -------
        pandas.Series
            A series with the predicted sky brightness for each row in the dataframe.
        """
        self.clf_r = joblib.load(BASE_DIR + '../data/sky_model/sky_model_r.pkl')
        self.clf_g = joblib.load(BASE_DIR + '../data/sky_model/sky_model_g.pkl')
        self.clf_i = joblib.load(BASE_DIR + '../data/sky_model/sky_model_i.pkl')

    def predict(self, df):
        """df is a dataframe with columns:
        mooonillf: 0-1
        moonalt: degrees
        moon_dist: degrees
        azimuth: degrees
        altitude: degrees
        sunalt: degrees
        filterkey: 1, 2, 3"""

        filter_ids = df['filter_id'].unique()
        assert(np.sum(filter_ids > 3) == 0)

        sky = pd.Series(np.nan, index=df.index, name='sky_brightness')
        wg = (df['filter_id'] == FILTER_NAME_TO_ID['g'])
        if np.sum(wg):
            sky[wg] = self.clf_g.predict(df[wg])
        wr = (df['filter_id'] == FILTER_NAME_TO_ID['r'])
        if np.sum(wr):

            sky[wr] = self.clf_r.predict(df[wr])
        wi = (df['filter_id'] == FILTER_NAME_TO_ID['i'])
        if np.sum(wi):
            sky[wi] = self.clf_i.predict(df[wi])

        return sky


class FakeSkyBrightness(object):
    """
    A class to simulate sky brightness predictions.

    Methods
    -------
    __init__():
        Initializes the FakeSkyBrightness object.
    
    predict(df):
        Predicts sky brightness for the given DataFrame.
        Parameters:
            df (pd.DataFrame): The input data for which sky brightness is to be predicted.
        Returns:
            pd.Series: A series with constant sky brightness value of 20 for each entry in the input DataFrame.
    """

    def __init__(self):
        pass

    def predict(self, df):
        y = np.ones(len(df)) * 20.
        return pd.Series(y, index=df.index, name='sky_brightness')


def train_sky_model(filter_name='r', df=None):
    """
    Train a sky brightness model using the specified filter and data.

    Parameters:
    filter_name (str): The name of the filter to use ('r', 'g', or 'i'). Default is 'r'.
    df (pandas.DataFrame, optional): The dataframe containing the data. If None, the data will be loaded from a default CSV file.

    Returns:
    sklearn.pipeline.Pipeline: The trained model pipeline.

    The function performs the following steps:
    1. Maps the filter name to a filter ID.
    2. Loads the data if not provided.
    3. Filters the data based on the filter ID.
    4. Converts negative moon illumination values to positive.
    5. Splits the data into training and testing sets.
    6. Standardizes the features.
    7. Trains an XGBoost regressor model.
    8. Prints the model score on the test set.
    9. Saves the trained model to a file.

    Note:
    - The function assumes that the data file is located at '../data/ptf-iptf_diq.csv.gz' relative to BASE_DIR.
    - The trained model is saved to '../data/sky_model/sky_model_{filter_name}.pkl' relative to BASE_DIR.
    """

    # PTF used 4 for i-band
    filterid_map = {'r': 2, 'g': 1, 'i': 4}

    if df is None:
        df = pd.read_csv(BASE_DIR + '../data/ptf-iptf_diq.csv.gz')
    # note that this is by pid, so there are multiple entries per image...

    df = df[df['filterkey'] == filterid_map[filter_name]].copy()

    # IPAC stores negative moonillf, but astroplan.moon_illumination does not
    df.loc[:, 'moonillf'] = np.abs(df['moonillf'])

    # returns dataframes!
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df, df['sky_brightness'], test_size=0.2)

    # don't really need to standardize for RF, but preprocessing is nice
    # preprocessing through sklearn_pandas raises a deprecation warning
    # from sklearn, so skip it.
    mapper = DataFrameMapper([
        (['moonillf'], preprocessing.StandardScaler()),
        (['moonalt'],   preprocessing.StandardScaler()),
        (['moon_dist'], preprocessing.StandardScaler()),
        (['azimuth'],  preprocessing.StandardScaler()),
        (['altitude'], preprocessing.StandardScaler()),
        (['sunalt'],   preprocessing.StandardScaler())])
    #('filterkey',  None)])

    clf = pipeline.Pipeline([
        ('featurize', mapper),
        ('xgb', xgb.XGBRegressor())])
    #('svr', svm.SVR(kernel='poly',degree=2))])
    #('knr', neighbors.KNeighborsRegressor(n_neighbors=15, weights='distance', algorithm='auto'))])
    #('lm', linear_model.BayesianRidge())])
    #('rf', ensemble.RandomForestRegressor(n_jobs=-1))])

    clf.fit(X_train, y_train.values.reshape(-1, 1))
    print(clf.score(X_test, y_test.values.reshape(-1, 1)))

    joblib.dump(clf, BASE_DIR + '../data/sky_model/sky_model_{}.pkl'.format(filter_name))

    return clf
