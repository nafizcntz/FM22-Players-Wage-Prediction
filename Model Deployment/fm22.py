import pandas as pd
import numpy as np
import joblib
import folium
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, \
    train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from folium import GeoJson
from folium.plugins import heat_map,marker_cluster,MarkerCluster
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor, Pool
from scipy.stats import pearsonr, stats,shapiro
def fm22_prediction(name):
    X_test_ = pd.read_csv(r"C:\Users\ezelb\OneDrive\Belgeler\GitHub\FM22-Players-Wage-Prediction\Model Deployment\Model_X_test.csv")
    index = X_test_[X_test_["Name"] == str(name)].index.tolist()[0]
    final_df = pd.read_csv(r"C:\Users\ezelb\OneDrive\Belgeler\GitHub\FM22-Players-Wage-Prediction\Model Deployment\Model_Deployment.csv")
    model = joblib.load(r'C:\Users\ezelb\OneDrive\Belgeler\GitHub\FM22-Players-Wage-Prediction\Model Deployment\voting_clf_2.pkl')
    y = final_df["Wages"]
    X = final_df.drop(["Wages","Name"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=17)
    return model.predict(X_test.iloc[index].values.reshape(1, -1))
#X_test_[X_test_["Name"] == "Mert Çetin"].index.tolist()[0]


























