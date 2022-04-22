# Görselleştirme:
# -Dünya haritası görselleştirmesi(folium)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, \
    train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from geopy.geocoders import Nominatim
from wordcloud import WordCloud
from folium import GeoJson
import folium
from folium.plugins import heat_map,marker_cluster

pd.set_option("display.max_columns",None)
pd.set_option("display.float_format",lambda x:"%.3f" %x)
pd.set_option("display.width",500)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
    return missing_df
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def outlier_plot(dataframe, numeric_cols,bol=3):
    fig, ax = plt.subplots(nrows=int(len(numeric_cols)/bol), ncols=bol, figsize=(10, 10))
    fig.tight_layout(pad=1.0)
    t = 0
    for i in range(int(len(numeric_cols)/bol)):
        for j in range(bol):
            sns.boxplot(x=dataframe[numeric_cols[t]], ax=ax[i, j])
            t += 1
    plt.show()
def outlier_replace(dataframe, numeric_cols, replace=False, lb_down=1.5, ub_up=1.5):
    lower_and_upper = {}
    for col in numeric_cols:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = 1.5 * (q3 - q1)

        lower_bound = q1 - iqr
        upper_bound = q3 + iqr

        lower_and_upper[col] = (lower_bound, upper_bound)
        if replace:
            dataframe.loc[(dataframe.loc[:, col] < lower_bound), col] = lower_bound * lb_down
            dataframe.loc[(dataframe.loc[:, col] > upper_bound), col] = upper_bound * ub_up
    print(lower_and_upper)
def plot_importance(model, features,save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(25,25 ))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False))
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('Best_Model_Feature_Importance.png')
    return feature_imp.sort_values("Value",ascending=False)

###################################
# Loading Datas
###################################


df_new = pd.read_csv("data/fm22_players_15k.csv")





# Baskılama işlemi yapmadık sadece wage değişeknindeki outlier datayı sildik
# çiftleme verileri silme
df_new = df_new[~df_new["Link"].duplicated()]


# Veri setinden bulunan "None" değerlerine boş değer ataması yaptık
df_new = df_new.replace(to_replace="None", value=np.nan)

# Release_Clause ---> Çok fazla boş değer bulunduğundan ötürü veri setinden çıkarttık.
# Diğer kolonlar model açısından bilgi içermiyeceğinden ötürü çıkarttık

df_new.drop(["Unnamed: 0",'Link',"Release_Clause","Img_Link","Unique_ID","Name"], axis=1,inplace=True)

#############################
# Boş değer doldurma
############################

# Hedef değişkende bulunan boş değerler silind
df_new = df_new[df_new["Wages"].notna()]

# Sell_value(Modeli kötü etkiliyebilir) ! Kötü bir futbolcuyu median'a göre doldurmuş olabilir
df_new.loc[df_new["Sell_Value"] == "Not for sale","Sell_Value"] = np.nan
df_new.loc[~df_new["Sell_Value"].isnull(),"Sell_Value"] = df_new[~df_new["Sell_Value"].isnull()]["Sell_Value"].str[2:].str.replace(",", "").astype(float)
df_new["Sell_Value"].fillna((df_new.loc[~df_new["Sell_Value"].isnull(),"Sell_Value"].median()),inplace=True)

missing_df=missing_values_table(df_new)
missing_df.head(40)


# Potential 22
 # İş bilgisine dayanarak(Wonderkid futbolcular) ve sitede bulunan bilgilendirmeler ile  Potensiyel gücü eksik olan veriler dolduruldu
df_new.loc[(df_new["Potential"].isnull()) & (df_new["Ability"] >= 70), "Potential"] = 90
df_new.loc[(df_new["Potential"].isnull()) & (df_new["Ability"] < 70), "Potential"] = 80

df_new[['Caps', 'Goals']] = df_new['Caps_Goals'].str.split('/', expand=True)
df_new.drop("Caps_Goals", axis=1, inplace=True)
df_new["Caps"] = df_new["Caps"].astype(int)
df_new["Goals"] = df_new["Goals"].astype(int)

df_new.dropna(inplace=True)

#########################
# Veri düzeltme
#########################

# Length
df_new["Length"] = df_new["Length"].str[:3]
df_new["Length"] = df_new["Length"].astype(int)



# Weight
df_new["Weight"] = df_new["Weight"].str[:2]
df_new["Weight"] = df_new["Weight"].astype(int)



#Wages
df_new["Wages"] = df_new["Wages"].str[1:-2].str.replace(",", "").astype(float)

# Contract_End
df_new["Contract_End"] = df_new["Contract_End"].apply(pd.to_datetime)
df_new['Contract_End_Year'] = df_new["Contract_End"].dt.year


# Oyuncuların pozisyon bilgisinin sınıflarını birleştirdik
df_new.loc[((df_new['Position'].str.contains("ST")) | (df_new['Position'].str.contains("AMR")) | (df_new['Position'].str.contains("AML"))), "Position"] = "Striker"
df_new.loc[((df_new['Position'].str.contains("DM")) | (df_new['Position'].str.contains("ML")) | (df_new['Position'].str.contains("MC")) | (df_new['Position'].str.contains("MR")) | (df_new['Position'].str.contains("AMC"))), "Position"] = "Midfield"
df_new.loc[((df_new['Position'].str.contains("DL")) | (df_new['Position'].str.contains("DR")) | (df_new['Position'].str.contains("DC")) | (df_new['Position'].str.contains("WBL")) | (df_new['Position'].str.contains("WBR"))), "Position"] = "Defenders"


cat_cols, num_cols, cat_but_car = grab_col_names(df_new)


#########################################
# Future Engineering
########################################
###################
#Değişken Üretme
###################


#####################
#Eksik değer analizi
######################
missing_values_table(df_new)
df_new.dropna(inplace=True)
df_new.shape
df_new.info()

################
# Encoding
################

cat_cols, num_cols, cat_but_car = grab_col_names(df_new)

# Nation ile CNation kolonlarında ortak değerler olduğundan ötürü aynı değerlerin
# atamsı labelencoder ile yapıldı.
labelencoder = LabelEncoder()
encoder_df = df_new[["Nation"]]
labelencoder.fit(encoder_df.stack().unique())
encoder_df['Nation'] = labelencoder.transform(encoder_df['Nation'])
df_new["Nation"] = encoder_df["Nation"]

############
encoder_df = df_new["Team"]
labelencoder.fit(encoder_df)
encoder_df['Team'] = labelencoder.transform(encoder_df)
df_new["Team"] = encoder_df["Team"]



# Label Encoder ile yapıldı
df_new = one_hot_encoder(df_new, ["Foot", "Position"], drop_first=True)
df_new.info(verbose=True)


final_df =df_new
#########################################
# Model Kurma
#########################################
y = final_df["Wages"]
X = final_df.drop(["Wages","Contract_End"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=17)


def models_(X_train, X_test, y_train, y_test,log=False):
    models = []
    models.append(('RF', RandomForestRegressor()))
    models.append(('GBM', GradientBoostingRegressor()))
    models.append(("XGBoost", XGBRegressor(objective='reg:squarederror')))
    models.append(("LightGBM", LGBMRegressor()))
    models.append(("CatBoost", CatBoostRegressor(verbose=False)))
    names = []
    rmse = []
    mae=[]
    if log:
        #log
        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse.append(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred))))
            names.append(name)
            mae.append(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)))
        tr_split = pd.DataFrame({'Name': names, 'RMSE': rmse,"MAE":mae})
        tr_split = tr_split.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
        print(tr_split,"\n")
        print(" Mean: ",np.expm1(y).mean(),"\n","Median: ",np.expm1(y).median(),"\n","Std: ",np.expm1(y).std())
    else:
        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            names.append(name)
            mae.append(mean_absolute_error(y_test, y_pred))
        tr_split = pd.DataFrame({'Name': names, 'RMSE': rmse,"MAE":mae})
        tr_split = tr_split.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
        print(tr_split,"\n")
        print(" Mean: ",y.mean(),"\n","Median: ",y.median(),"\n","Std: ",y.std())
        sorted_models = []
        for name in tr_split["Name"]:
            for i, col in enumerate(models):
                if col[0] == name:
                    sorted_models.append(col)

    return "LightGBM",sorted_models,tr_split

def best_model(X_train, X_test, y_train, y_test,plot_1=False,plot_2=False):
    model_name,models,tr_split = models_(X_train, X_test, y_train, y_test)
    for name, model in models:
        if name == model_name:
            modelfit = model.fit(X_train, y_train)
            y_pred = modelfit.predict(X_test)
            print("\nTest RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
            # Train Rmse
            y_pred_tr = modelfit.predict(X_train)
            print("Train RMSE: ", np.sqrt(mean_squared_error(y_train, y_pred_tr)))
            # Ekstra
            print("\nTest Score: ",modelfit.score(X_test, y_test),"\nTrain Score:",modelfit.score(X_train, y_train))
            # yüzdesel rmse
            rmspe = np.sqrt(np.mean(np.square(((y_test - y_pred) / y_test)), axis=0))
            print("\nrmspe: ",rmspe )

            # Cross Validation
            cv_results = cross_validate(modelfit, X_train, y_train, cv=10,
                                        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
            print("CV RMSE   : ",-(cv_results['test_neg_root_mean_squared_error'].mean()))
            print("CV MAE    : ",-(cv_results['test_neg_mean_absolute_error'].mean()))
            print("CV R-KARE :",(cv_results['test_r2'].mean()))
            df_feature = plot_importance(model, X_train, save=True)
            if plot_1:
                fig, ax = plt.subplots(figsize=(15, 10))
                plt.suptitle('MODELLERİN KARŞILAŞTIRILMASI', fontsize=20)
                plt.yticks(fontsize=13)
                plt.xticks(fontsize=13)
                plt.plot(tr_split['Name'], tr_split['RMSE'], label="RMSE")
                plt.plot(tr_split['Name'], tr_split['MAE'], label="MAE")
                plt.legend()
                plt.show()
                plt.savefig("visualization/Model_RMSE_MAE.png")
            if plot_2:
                identity_line = np.linspace(max(min(y_pred), min(y_test)),
                                            min(max(y_pred), max(y_test)))
                plt.figure(figsize=(10, 10))
                plt.scatter(x=y_pred, y=y_test, alpha=0.2)
                plt.plot(identity_line, identity_line, color="red", linestyle="dashed", linewidth=3.0)
                plt.show()
                plt.savefig("visualization/Gerçek_tahmin_Dağılımı.png")
            return df_feature,modelfit,models

df_feature_model,modelfit,models = best_model(X_train, X_test, y_train, y_test)

# CV RMSE   :  3729.0600339496295
# CV MAE    :  2183.8718166773024
# CV R-KARE : 0.815188708820536



# Wages(Base Model)
# Wages
#1)
# Mean:  5590.081766917293
# Median:  3740.0
# Std:  5235.466436711407
#2)
# Mean:  8078.075976457999
# Median:  4820.0
# Std:  8623.365117077497
# 3) --> PCA 11 column

#        Name     RMSE|        Name     RMSE|       Name     RMSE|
# 0  CatBoost 3447.107| 0  LightGBM 3606.873|0  CatBoost 5122.401|
# 1  LightGBM 3546.376| 1  CatBoost 3629.678|1  LightGBM 5232.242|
# 2       GBM 3588.416| 2   XGBoost 3785.523|2        RF 5240.202|
# 3        RF 3630.061| 3       GBM 3852.540|3   XGBoost 5378.897|
# 4   XGBoost 3657.617| 4        RF 3857.140|4       GBM 5416.904|



#####################################
# Feature Importance New Model
####################################
importance_col = df_feature_model[df_feature_model["Value"]!=0]["Feature"].tolist()
y = final_df["Wages"]
X = final_df[importance_col]
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=17)
best_model(X_train, X_test, y_train, y_test,plot=True)



# Hiperparametre Optimizasyonu
modelfit.get_params()
model_params = {'max_depth': range(-1, 11),
                   "min_child_samples": range(10, 30, 5),
                   'n_estimators': range(100,1000,100),
                   'learning_rate': [0.1,0.2,0.3,0.4,0.5],
                   "colsample_bytree": [0.9, 0.8, 1]}
model_best_grid = GridSearchCV(modelfit, model_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X_train, y_train)
best_params = {'max_depth': 3,
               "min_child_samples": 15,
               'n_estimators': 100,
               'learning_rate': 0.1,
               "colsample_bytree": 0.8}

model_best_grid.best_params_
model_best_grid.best_score_

model_final = modelfit.set_params(**model_best_grid.best_params_, random_state=17).fit(X_train, y_train)
y_pred = model_final.predict(X_test)
print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
cv_results = cross_validate(model_final, X_train, y_train, cv=20, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
cv_results['test_neg_root_mean_squared_error'].mean()
cv_results['test_neg_mean_absolute_error'].mean()
cv_results['test_r2'].mean()

from sklearn.ensemble import VotingRegressor

def voting_regression(models, X_train, y_train):
    print("Voting Regression...")
    voting_clf = VotingRegressor(estimators=[(models[0][0], models[0][1]), (models[1][0], models[1][1]),
                                              (models[2][0], models[2][1])]).fit(X_train, y_train)

    cv_results = cross_validate(voting_clf, X_train, y_train, cv=5,
                                scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
    print("VR RMSE   : ", -(cv_results['test_neg_root_mean_squared_error'].mean()))
    print("VR MAE    : ", -(cv_results['test_neg_mean_absolute_error'].mean()))
    print("VR R-KARE :", (cv_results['test_r2'].mean()))
    return voting_clf

voting_regression(models,X_train,y_train)

# VR RMSE   :  3714.203160096844
# VR MAE    :  2154.799428883318
# VR R-KARE : 0.8168473801393296

###############################################
# Catboost ile overfitting önleme
from catboost import CatBoostRegressor, Pool
class RmseMetric(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approx[i] - target[i])**2)

        return error_sum, weight_sum

y = df_new["Wages"]
X = df_new.drop(["Wages","Sell_Value"], axis=1)
#X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.3)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)

X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)

train_pool = Pool(X, y)
eval_pool = Pool(X_valid, y_valid)

model = CatBoostRegressor(iterations=2500, learning_rate=0.1, eval_metric=RmseMetric())

modelfit = model.fit(X_train, y_train, eval_set=eval_pool, early_stopping_rounds=10)

y_pred = modelfit.predict(X_test)
print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Train Rmse
y_pred_tr = modelfit.predict(X_train)
print("Train RMSE: ", np.sqrt(mean_squared_error(y_train, y_pred_tr)))



def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(25, 25))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    return feature_imp

plot_importance(modelfit, X_train)

