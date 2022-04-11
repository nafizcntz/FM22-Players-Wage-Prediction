# Görselleştirme:
# -Dünya haritası görselleştirmesi(folium)

import pandas as pd
import numpy as np
import joblib
import folium
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve, \
    train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from geopy.geocoders import Nominatim
from wordcloud import WordCloud
from folium import GeoJson
from folium.plugins import heat_map,marker_cluster
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor, Pool
from scipy.stats import pearsonr, stats,shapiro

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
def load_data():
    df_clubs = pd.read_csv("data/fm22_clubs_3k.csv")
    df_new = pd.read_csv("data/fm22_players_15k.csv")
    df_fm21 = pd.read_csv("data/fm21_players_15k.csv")
    df_clubs_fm21 = pd.read_csv("data/fm21_clubs_3k.csv")
    return df_clubs,df_new,df_fm21,df_clubs_fm21


##################################
# EDA
###################################
df_new = load_data()
check_df(df_new)
na_col = missing_values_table(df_new)
missing_values_table(df_new).head(20)
cat_cols, num_cols, cat_but_car = grab_col_names(df_new)
# Outlier Görselleştirme
outlier_plot(df_new,num_cols)

def data_prep(df_new,df_clubs,df_fm21,df_clubs_fm21):
    ########################################
    # Veri Ön İşleme
    ########################################
    # Külüp datasındaki kolonları ana data setine daha sonra doldurmak üzere
    # boş değer ataması yaparak ekledik
    #  Oyuncuların kuluplerine bakarak club datasetindeki bilgileri oyuncu özelinde
    # atama işlemi yaptık
    df_new[df_clubs.columns] = np.nan
    for i,col in enumerate(df_new["Team"]):
        for j,col_club in enumerate(df_clubs["CName"]):
            if col == col_club:
                df_new.iloc[i,54:] = df_clubs.loc[j]
                break

    # Fm21'e club datasını ekleme
    df_fm21[df_clubs_fm21.columns] = np.nan
    for i,col in enumerate(df_fm21["Team"]):
        for j,col_club in enumerate(df_clubs_fm21["CName"]):
            if col == col_club:
                df_fm21.iloc[i,54:] = df_clubs_fm21.loc[j]
                break
    # FM21 ekleme
    df_fm21.drop('Unnamed: 0',axis=1,inplace=True)
    df_new[df_fm21.columns[4:]+ "_fm21"] = np.nan
    lst =[i for i in df_new.columns if "fm21" in i ]

    for i,col in enumerate(df_new["Name"]):
        for j,col_fm21 in enumerate(df_fm21["Name"]):
            if col == col_fm21:
                df_new.loc[i,lst] = df_fm21.iloc[j,4:].values
                break

    # çiftleme verileri silme
    df_new = df_new[~df_new["Link"].duplicated()]


    # Veri setinden bulunan "None" değerlerine boş değer ataması yaptık
    df_new = df_new.replace(to_replace="None", value=np.nan)

    # Release_Clause ---> Çok fazla boş değer bulunduğundan ötürü veri setinden çıkarttık.
    # Diğer kolonlar model açısından bilgi içermiyeceğinden ötürü çıkarttık
    df_name_Cname = df_new[["Name","CName"]]
    df_new.drop(["Unnamed: 0",'Link',"Release_Clause","CStatus_fm21","CName","CLink","CName_fm21","CLink_fm21","CStatus","Unique_ID_fm21",'Unique_ID',"Name","Release_Clause_fm21"], axis=1,inplace=True)


    #############################
    # Boş değer doldurma
    ############################

    # Hedef değişkende bulunan boş değerler silindi
    df_new = df_new[df_new["Wages"].notna()]

    # Sell_value(Modeli kötü etkiliyebilir) ! Kötü bir futbolcuyu median'a göre doldurmuş olabilir
    df_new.loc[df_new["Sell_Value"] == "Not for sale","Sell_Value"] = np.nan
    df_new.loc[~df_new["Sell_Value"].isnull(),"Sell_Value"] = df_new[~df_new["Sell_Value"].isnull()]["Sell_Value"].str[2:].str.replace(",", "").astype(float)
    df_new["Sell_Value"].fillna((df_new.loc[~df_new["Sell_Value"].isnull(),"Sell_Value"].median()),inplace=True)


    # Potential 22
     # İş bilgisine dayanarak(Wonderkid futbolcular) ve sitede bulunan bilgilendirmeler ile  Potensiyel gücü eksik olan veriler dolduruldu
    df_new.loc[(df_new["Potential"].isnull()) & (df_new["Ability"] >= 70), "Potential"] = 90
    df_new.loc[(df_new["Potential"].isnull()) & (df_new["Ability"] < 70), "Potential"] = 80

    # Potential 21
    df_new.loc[(df_new["Potential_fm21"].isnull()) & (df_new["Ability_fm21"] >= 70), "Potential_fm21"] = 90
    df_new.loc[(df_new["Potential_fm21"].isnull()) & (df_new["Ability_fm21"] < 70), "Potential_fm21"] = 80

    # Fm21 verisi olmayanları sildik
    df_new = df_new[~df_new["Ability_fm21"].isnull()]

    #Geri Kalanları sildik
    df_new.dropna(inplace=True)


    #########################
    # Veri düzeltme
    #########################

    # Length
    df_new["Length"] = df_new["Length"].str[:3]
    df_new["Length"] = df_new["Length"].astype(int)
    # Length fm21
    df_new["Length_fm21"] = df_new["Length_fm21"].str[:3]
    df_new["Length_fm21"] = df_new["Length_fm21"].astype(int)


    # Weight
    df_new["Weight"] = df_new["Weight"].str[:2]
    df_new["Weight"] = df_new["Weight"].astype(int)
    # Weight_fm21
    df_new["Weight_fm21"] = df_new["Weight_fm21"].str[:2]
    df_new["Weight_fm21"] = df_new["Weight_fm21"].astype(int)


    #Wages
    df_new["Wages"] = df_new["Wages"].str[1:-2].str.replace(",", "").astype(float)

    #Wages_fm21
    df_new["Wages_fm21"] = df_new["Wages_fm21"].str[1:-2].str.replace(",", "").astype(float)

    # Contract_End
    df_new["Contract_End"] = df_new["Contract_End"].apply(pd.to_datetime)
    df_new['Contract_End_Year'] = df_new["Contract_End"].dt.year

    # Contract_End_21
    df_new["Contract_End_fm21"] = df_new["Contract_End_fm21"].apply(pd.to_datetime)
    df_new['Contract_End_Year_fm21'] = df_new["Contract_End_fm21"].dt.year



    # Sell_Value_fm21
    df_new = df_new.loc[df_new["Sell_Value_fm21"] != "Not for sale"]
    df_new["Sell_Value_fm21"] = df_new["Sell_Value_fm21"].str[2:].str.replace(",", "").astype(float)


    # CBalance
    df_new.loc[df_new["CBalance"].str.contains("K") == True,"CBalance"] = df_new[df_new["CBalance"].str.contains("K") == True]["CBalance"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CBalance"].str.contains("M") == True,"CBalance"] = df_new[df_new["CBalance"].str.contains("M") == True]["CBalance"].str[1:-1].astype(float) * (10**6)
    df_new["CBalance"] = df_new["CBalance"].astype(float)

    # CBalance_fm21
    df_new.loc[df_new["CBalance_fm21"].str.contains("K") == True,"CBalance_fm21"] = df_new[df_new["CBalance_fm21"].str.contains("K") == True]["CBalance_fm21"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CBalance_fm21"].str.contains("M") == True,"CBalance_fm21"] = df_new[df_new["CBalance_fm21"].str.contains("M") == True]["CBalance_fm21"].str[1:-1].astype(float) * (10**6)
    df_new["CBalance_fm21"] = df_new["CBalance_fm21"].astype(float)


    # CTransfer_Budget
    df_new.loc[df_new["CTransfer_Budget"].str.contains("K") == True,"CTransfer_Budget"] = df_new[df_new["CTransfer_Budget"].str.contains("K") == True]["CTransfer_Budget"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CTransfer_Budget"].str.contains("M") == True,"CTransfer_Budget"] = df_new[df_new["CTransfer_Budget"].str.contains("M") == True]["CTransfer_Budget"].str[1:-1].astype(float) * (10**6)
    df_new["CTransfer_Budget"] = df_new["CTransfer_Budget"].astype(float)

    # CTransfer_Budget_fm21
    df_new.loc[df_new["CTransfer_Budget_fm21"].str.contains("K") == True,"CTransfer_Budget_fm21"] = df_new[df_new["CTransfer_Budget_fm21"].str.contains("K") == True]["CTransfer_Budget_fm21"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CTransfer_Budget_fm21"].str.contains("M") == True,"CTransfer_Budget_fm21"] = df_new[df_new["CTransfer_Budget_fm21"].str.contains("M") == True]["CTransfer_Budget_fm21"].str[1:-1].astype(float) * (10**6)
    df_new["CTransfer_Budget_fm21"] = df_new["CTransfer_Budget_fm21"].astype(float)


    # CTotal_Wages
    df_new["CTotal_Wages"] = df_new["CTotal_Wages"].str[:-2]
    df_new["CTotal_Wages"] = df_new["CTotal_Wages"].str.strip()
    df_new.loc[df_new["CTotal_Wages"].str.contains("K") == True,"CTotal_Wages"] = df_new[df_new["CTotal_Wages"].str.contains("K") == True]["CTotal_Wages"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CTotal_Wages"].str.contains("M") == True,"CTotal_Wages"] = df_new[df_new["CTotal_Wages"].str.contains("M") == True]["CTotal_Wages"].str[1:-1].astype(float) * (10**6)
    df_new["CTotal_Wages"] = df_new["CTotal_Wages"].astype(float)

    # CTotal_Wages_fm21
    df_new["CTotal_Wages_fm21"] = df_new["CTotal_Wages_fm21"].str[:-2]
    df_new["CTotal_Wages_fm21"] = df_new["CTotal_Wages_fm21"].str.strip()
    df_new.loc[df_new["CTotal_Wages_fm21"].str.contains("K") == True,"CTotal_Wages_fm21"] = df_new[df_new["CTotal_Wages_fm21"].str.contains("K") == True]["CTotal_Wages_fm21"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CTotal_Wages_fm21"].str.contains("M") == True,"CTotal_Wages_fm21"] = df_new[df_new["CTotal_Wages_fm21"].str.contains("M") == True]["CTotal_Wages_fm21"].str[1:-1].astype(float) * (10**6)
    df_new["CTotal_Wages_fm21"] = df_new["CTotal_Wages_fm21"].astype(float)

    # CRemaining_Wages
    df_new["CRemaining_Wages"] = df_new["CRemaining_Wages"].str[:-2]
    df_new["CRemaining_Wages"] = df_new["CRemaining_Wages"].str.strip()
    df_new.loc[df_new["CRemaining_Wages"].str.contains("K") == True,"CRemaining_Wages"] = df_new[df_new["CRemaining_Wages"].str.contains("K") == True]["CRemaining_Wages"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CRemaining_Wages"].str.contains("M") == True,"CRemaining_Wages"] = df_new[df_new["CRemaining_Wages"].str.contains("M") == True]["CRemaining_Wages"].str[1:-1].astype(float) * (10**6)
    df_new["CRemaining_Wages"] = df_new["CRemaining_Wages"].astype(float)

    # CRemaining_Wages_fm21
    df_new["CRemaining_Wages_fm21"] = df_new["CRemaining_Wages_fm21"].str[:-2]
    df_new["CRemaining_Wages_fm21"] = df_new["CRemaining_Wages_fm21"].str.strip()
    df_new.loc[df_new["CRemaining_Wages_fm21"].str.contains("K") == True,"CRemaining_Wages_fm21"] = df_new[df_new["CRemaining_Wages_fm21"].str.contains("K") == True]["CRemaining_Wages_fm21"].str[1:-1].astype(float) * 1000
    df_new.loc[df_new["CRemaining_Wages_fm21"].str.contains("M") == True,"CRemaining_Wages_fm21"] = df_new[df_new["CRemaining_Wages_fm21"].str.contains("M") == True]["CRemaining_Wages_fm21"].str[1:-1].astype(float) * (10**6)
    df_new["CRemaining_Wages_fm21"] = df_new["CRemaining_Wages_fm21"].astype(float)


    # CFounded
    df_new["CFounded"] = pd.to_numeric(df_new["CFounded"],errors='coerce')

    # CFounded_fm21
    df_new["CFounded_fm21"] = pd.to_numeric(df_new["CFounded_fm21"],errors='coerce')

    # CMost_Talented_XI
    df_new["CMost_Talented_XI"] = df_new["CMost_Talented_XI"].astype(int)

    # CMost_Talented_XI_fm21
    df_new = df_new[~(df_new["CMost_Talented_XI_fm21"]== "NAN")]
    df_new["CMost_Talented_XI_fm21"] = df_new["CMost_Talented_XI_fm21"].astype(int)

    #CBest_XI_fm21
    df_new["CBest_XI_fm21"] = df_new["CBest_XI_fm21"].astype(int)

    # Oyuncuların pozisyon bilgisinin sınıflarını birleştirdik
    df_new.loc[((df_new['Position'].str.contains("ST")) | (df_new['Position'].str.contains("AMR")) | (df_new['Position'].str.contains("AML"))), "Position"] = "Striker"
    df_new.loc[((df_new['Position'].str.contains("DM")) | (df_new['Position'].str.contains("ML")) | (df_new['Position'].str.contains("MC")) | (df_new['Position'].str.contains("MR")) | (df_new['Position'].str.contains("AMC"))), "Position"] = "Midfield"
    df_new.loc[((df_new['Position'].str.contains("DL")) | (df_new['Position'].str.contains("DR")) | (df_new['Position'].str.contains("DC")) | (df_new['Position'].str.contains("WBL")) | (df_new['Position'].str.contains("WBR"))), "Position"] = "Defenders"

    ############################
    # Outlier
    ############################
    # Filtreleme
    df_new = df_new.loc[df_new["Wages"] < 45270.0]
    return df_new,df_name_Cname
def visualization(df_new,df_name_Cname):
    ###################################
    # Veri Görselleştirme
    ###########################
    # Görselleştirme için name ve Cname'i başka bir veri setinden tutma
    df_new,df_name_Cname = data_prep(df_new)
    df_name_Cname=df_name_Cname.loc[df_new.index]
    df_new.reset_index(drop=True,inplace=True)
    df_name_Cname.reset_index(drop=True,inplace=True)

    # Potential & Wage
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x =df_new['Potential'], y = df_new['Wages'])
    plt.xlabel("Potential")
    plt.ylabel("Wage")
    plt.title("Potential & Wage", fontsize = 18)
    plt.savefig("visualization/Potential&Wage_Dağılımı.png")
    plt.show()

    # Potential & Sell_Value
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x =df_new['Potential'], y = df_new['Sell_Value'])
    plt.xlabel("Potential")
    plt.ylabel("Sell_Value")
    plt.title("Potential & Sell_Value", fontsize = 18)
    plt.savefig("visualization/Potential&Sell_Value_Dağılımı.png")
    plt.show()


    # Ayak kırılımında potential&Wage dağılımı
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x =df_new['Potential'], y = df_new['Wages'], hue = df_new['Foot'])
    plt.xlabel("Potential")
    plt.ylabel("Wage")
    plt.title("Foot & Potential & Wage", fontsize = 18)
    plt.savefig("visualization/Ayak_Potential&Wage_dağılımı.png")
    plt.show()

    # Wordcloud
    text = " ".join(i for i in df_new.Nation)
    wordcloud = WordCloud(collocations=False).generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("visualization/Nation_WordCloud.png")
    plt.show()


    # # Data Dağılım (['Length', 'Weight', 'Caps_Goals', 'Sell_Value', 'Wages'])
    # sns.pairplot(df_new[df_new.columns.tolist()[7:12]])
    # sns.set(style="ticks", color_codes=True)
    # plt.show()

    #Wage Sell_Value Dağılımları
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7))
    sns.kdeplot(df_new["Wages"],shade=True,ax=ax[0])
    sns.kdeplot(df_new["Sell_Value"],shade=True,ax=ax[1])
    plt.savefig("visualization/Wage_and_Sell_Value_Dağılımları.png")
    plt.show()

    # Dünya haritası ile görselleştirme
    df_geo = df_new[["CNation","CCity"]]
    df_geo["Long"] = np.nan
    df_geo["Lat"] = np.nan
    geolocator = Nominatim(user_agent="my_user_agent")
    lst_nation = df_geo["CNation"].value_counts().index.tolist()
    lst_city = df_geo["CCity"].value_counts().index.tolist()
    dict_nation={}
    for i in lst_city:
        loc = geolocator.geocode(i)
        if loc:
            dict_nation.update({i:[loc.longitude,loc.latitude]})
        else:
            print(i)
            dict_nation.update({i:[37.1833,67.3667]})
    df_geo["Potential"] = df_new["Potential"]
    df_geo["Potential_Mean"] = np.nan
    for i in lst_city :
        df_geo.loc[df_geo["CCity"] == i, "Long"] = dict_nation[i][0]
        df_geo.loc[df_geo["CCity"] == i, "Lat"] = dict_nation[i][1]

    for j in lst_nation:
        df_geo.loc[df_geo["CNation"] == j, "Potential_Mean"] = df_geo[df_geo["CNation"] == j]["Potential"].mean()

    df_geo[["Name","CName"]]= df_name_Cname
    df_geo[["Ability","Age","Foot","Position","Caps_Goals","Length","Weight","Nation","Wages"]]  =df_new[["Ability","Age","Foot","Position","Caps_Goals","Length","Weight","Nation","Wages"]]

    geo=r"archive/countries.geojson"
    file = open(geo, encoding="utf8")
    text = file.read()

    # Futbolcu potansiyellerine göre dağılmını map üzerinde gösterilmesi
    m = folium.Map([42 ,29],tiles="Cartodb Positron", zoom_start=5,width="%100",height="%100")
    folium.Choropleth(
        geo_data=text,
        data=df_geo,
        columns=['CNation', 'Potential_Mean'],
        legend_name='Oynadıkları liglere göre Potansiyel Oyuncu Dağılımı',
        key_on='feature.properties.ADMIN'
        ).add_to(m)
    m.save('visualization/Potensiyel_ortalamasına_göre_ülke_dağılımı.html')


    # Futbolcuların ülkelere göre dağılımını HeatMap olarak  gösterilmesi
    m=folium.Map(location=[40,32],tiles="OpenStreetMap",zoom_start=7)
    folium.plugins.HeatMap(zip(df_geo["Lat"],df_geo["Long"])).add_to(m)
    m.save('visualization/Nation_Heatmap.html')

    # Futbolcuların şehirlere göre göre dağılımını MarkerCluster gösterilmesi
    m=folium.Map(location=[40,32],tiles="OpenStreetMap",zoom_start=7)
    folium.plugins.MarkerCluster(zip(df_geo["Lat"],df_geo["Long"])).add_to(m)
    m.save('visualization/Nation_MarkerCluster.html')

    # Oyuncuların oyanığı takımlara göre dağılımı ve bilgilerinin gösterilmesi
    m3=folium.Map(location=[40,32],tiles="OpenStreetMap",zoom_start=7)
    for i in df_geo.index:
        iframe = folium.IFrame("<font face='Comic Sans MS'  color='#143F6B'>" +
                                '<h3><b> Name: </b></font>' + str(df_geo.loc[i,'Name']) + '</h3><br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Nation: </b></font>' + str(df_geo.loc[i, 'Nation']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Ability: </b></font>' + str(df_geo.loc[i, 'Ability']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Potential: </b></font>' + str(df_geo.loc[i, 'Potential']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Age: </b></font>' + str(df_geo.loc[i,'Age']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Position: </b></font>' + str(df_geo.loc[i, 'Position']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Foot: </b></font>' + str(df_geo.loc[i, 'Foot']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Length: </b></font>' + str(df_geo.loc[i, 'Length']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Weight: </b></font>' + str(df_geo.loc[i, 'Weight']) + '<br>' +
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Caps_Goals: </b></font>' + str(df_geo.loc[i, 'Caps_Goals'])+ '<br>'+
                               "<font face='Comic Sans MS'  color='#143F6B'>" +
                               '<b>Wages: </b></font>' + str(df_geo.loc[i, 'Wages']))
        popup = folium.Popup(iframe, min_width=300, max_width=300)
        lat=df_geo.loc[i,"Lat"]+np.random.uniform(0.1, 10**(-20))-0.00005
        long=df_geo.loc[i,"Long"]+np.random.uniform(0.1, 10**(-20))-0.00005
        folium.Marker(location=[lat, long],popup=popup).add_to(m3)
    m3.save('visualization/m3.html')
    ######################
    # Korelasyon
    ###################
    # Heatmap (Sınırlama yapılmalı)
    plt.figure(figsize=(30, 30))
    matrix = np.triu(df_new.corr())
    sns.heatmap(df_new.corr(), xticklabels=df_new.corr().columns, yticklabels=df_new.corr().columns, annot=True, mask=matrix)
    plt.show()
def corr_anlys(df_new):
    # Korelasyon Anlamlılığı
    df_new.columns
    cols = df_new.corr()["Wages"].sort_values(ascending=False).index
    def Shapiro_app(df,cols):
        for i in cols:
            test_istatistigi, p_value = shapiro(df[i])
            if p_value < 0.05:
                print("p-degeri < 0.05 oldugundan H0 red edilemez")
                print(str(i) + " degiskeni normal bir dagilima sahip degildir")
                print("-------------------------------------------------------")
            elif p_value > 0.05:

                print("p-degeri > 0.05 oldugundan H0 red edilir")
                print(str(i) + " degiskeni normal bir dagilima sahiptir")
                print("-------------------------------------------------------")

    Shapiro_app(df_new,cols)
    # Bazi degiskenler normal dagilima sahip iken bazilarinin olmadigini goruyoruz. Bu durumda normal dagilima sahip olanlar icin PERASON olmayanlar icin SPEARMAN korelasyon kat sayisi testini uygulayabiliriz.
    # Hipotez;
    # H0: p=0
    # Hs: p>0
    # şeklinde kurulur. Korelasyon katsayısının anlamlılığı t testi kullanılarak, n-2 serbestlik derecesine göre aşağıdaki işlemler dahilinde verilen anlamlılık düzeyine göre test edilir.
    corr, p = stats.spearmanr(df_new["Wages"], df_new["Long_Throws"])
    print('Spearman correlation: %.3f, P-value score: %.3f' % (corr,p))
def feature_eng(df_new):

    #########################################
    # Feature Engineering
    ########################################
    cat_cols, num_cols, cat_but_car = grab_col_names(df_new)
    ###################
    #Değişken Üretme
    ###################
    # Futbolcuların sözleşme bitiş tarihini kullanarak yeni değişkenler oluşturuldu.
    today_date = dt.datetime(2022, 1, 1)
    df_new['Contrat_end_month'] = df_new["Contract_End"].dt.month
    df_new['Contrat_end_day'] = df_new["Contract_End"].dt.day
    df_new['Contrat_end_year'] = df_new["Contract_End"].dt.year
    df_new["Contrat_end_left_days"] = (df_new["Contract_End"]-today_date).dt.days
    df_new["Contrat_end_left_year"] = (df_new["Contract_End"].dt.year-today_date.year)
    df_new["Contrat_end_left_month"] = (df_new["Contract_End"]-today_date)/np.timedelta64(1,"M")
    df_new.drop("Contract_End",axis=1,inplace=True)

    # fm21
    today_date = dt.datetime(2021, 1, 1)
    df_new['Contrat_end_month_fm21'] = df_new["Contract_End_fm21"].dt.month
    df_new['Contrat_end_day_fm21'] = df_new["Contract_End_fm21"].dt.day
    df_new['Contrat_end_year_fm21'] = df_new["Contract_End_fm21"].dt.year
    df_new["Contrat_end_left_days_fm21"] = (df_new["Contract_End_fm21"]-today_date).dt.days
    df_new["Contrat_end_left_year_fm21"] = (df_new["Contract_End_fm21"].dt.year-today_date.year)
    df_new["Contrat_end_left_month_fm21"] = (df_new["Contract_End_fm21"]-today_date)/np.timedelta64(1,"M")
    df_new.drop("Contract_End_fm21",axis=1,inplace=True)


    # Oyuncuların yaşlarına ve potensiyel güçlerine göre sınıfladırma yapılarak yeni değişken üretildi
    df_new["Age_Potential_Seg"] = ""
    df_new.loc[(df_new["Age"] <= 20) & (df_new["Potential"] >= 80), "Age_Potential_Seg"] = "Wonderkid"
    df_new.loc[(df_new["Age"] <= 20) & (df_new["Potential"] < 80), "Age_Potential_Seg"] = "Tecrübesiz"
    df_new.loc[(df_new["Age"] > 20) & (25 >= df_new["Age"]) & (df_new["Potential"] >= 80), "Age_Potential_Seg"] = "Star_Candidate"
    df_new.loc[(df_new["Age"] > 20) & (25 >= df_new["Age"]) & (df_new["Potential"] < 80), "Age_Potential_Seg"] = "Developers"
    df_new.loc[(df_new["Age"] > 25) & (35 >= df_new["Age"]) & (df_new["Potential"] >= 80), "Age_Potential_Seg"] = "Star"
    df_new.loc[(df_new["Age"] > 25) & (35 >= df_new["Age"]) & (df_new["Potential"] < 80), "Age_Potential_Seg"] = "Star-"
    df_new.loc[(35 < df_new["Age"]) & (df_new["Potential"] >= 80), "Age_Potential_Seg"] = "Star+"
    df_new.loc[(35 < df_new["Age"]) & (df_new["Potential"] < 80), "Age_Potential_Seg"] = "Çöp"

    #  Yeni değişkenler
    df_new["Ability_Potential"] = df_new["Ability"] * df_new["Potential"]
    df_new["New_Most_Best"] = df_new["CBest_XI"] - df_new["CMost_Talented_XI"]
    df_new["New_Rep_Best_Tal"] = df_new["CBest_XI"] * df_new["CMost_Talented_XI"] * df_new["CReputation"]
    df_new["New_Tack_Mark"] = df_new["Tackling"] + df_new["Marking"]
    df_new["New_Pos_Mark"] = df_new["Positioning"] * df_new["Marking"]
    df_new["New_Jump_Leng"] = df_new["Length"] / df_new["Jumping_Reach"]

    #FM22 - FM21
    df_new["Ability_Change"] = df_new["Ability"] - df_new["Ability_fm21"]
    df_new["Potential_Change"] = df_new["Potential"] - df_new["Potential_fm21"]
    df_new["Sell_Value_Change"] = df_new["Sell_Value"] - df_new["Sell_Value_fm21"]

    # Kategori
    # Hucüm futbolcularını sınıflara ayırma
    dff = df_new[num_cols[8:44]]
    dff["Position"] = df_new["Position"]
    dff_1 = dff[dff["Position"]=="Midfield"].iloc[:,:-1]
    sc = MinMaxScaler((0, 1))
    dff_1_scale = sc.fit_transform(dff_1)
    kmeans = KMeans(n_clusters=4, random_state=17).fit(dff_1_scale)
    dff["Clusters"] = np.nan
    dff.loc[dff[dff["Position"] == "Midfield"].index,"Clusters"] = kmeans.labels_
    dff_2 = dff[dff["Position"]=="Striker"].iloc[:,:-2]
    dff_2_scale = sc.fit_transform(dff_2)
    kmeans = KMeans(n_clusters=4, random_state=17).fit(dff_2_scale)
    kmeans.labels_ = kmeans.labels_+ 4
    dff.loc[dff[dff["Position"] == "Striker"].index,"Clusters"] = kmeans.labels_
    dff_3 = dff[dff["Position"]=="Defenders"].iloc[:,:-2]
    dff_3_scale = sc.fit_transform(dff_3)
    kmeans = KMeans(n_clusters=4, random_state=17).fit(dff_3_scale)
    kmeans.labels_ = kmeans.labels_+ 8
    dff.loc[dff[dff["Position"] == "Defenders"].index,"Clusters"] = kmeans.labels_
    dff["Clusters"].astype(int).astype(str)
    df_new["Kategori"] = dff["Clusters"].astype(int).astype(str)



    # Caps değişkeni iki değişkene ayırıldı(veri önişlemeye alınabilir)
    df_new[['Caps', 'Goals']] = df_new['Caps_Goals'].str.split('/', expand=True)
    df_new.drop("Caps_Goals", axis=1, inplace=True)
    df_new["Caps"] = df_new["Caps"].astype(int)
    df_new["Goals"] = df_new["Goals"].astype(int)

    df_new[['Caps_fm21', 'Goals_fm21']] = df_new['Caps_Goals_fm21'].str.split('/', expand=True)
    df_new.drop("Caps_Goals_fm21", axis=1, inplace=True)
    df_new["Caps_fm21"] = df_new["Caps_fm21"].astype(int)
    df_new["Goals_fm21"] = df_new["Goals_fm21"].astype(int)


    ################
    # Encoding
    ################
    cat_cols, num_cols, cat_but_car = grab_col_names(df_new)
    # Nation ile CNation kolonlarında ortak değerler olduğundan ötürü aynı değerlerin
    # atamsı labelencoder ile yapıldı.
    labelencoder = LabelEncoder()
    encoder_df = df_new[["Nation", "CNation","CNation_fm21"]]
    labelencoder.fit(encoder_df.stack().unique())
    encoder_df['Nation'] = labelencoder.transform(encoder_df['Nation'])
    encoder_df['CNation'] = labelencoder.transform(encoder_df['CNation'])
    encoder_df['CNation_fm21'] = labelencoder.transform(encoder_df['CNation_fm21'])
    df_new["Nation"] = encoder_df["Nation"]
    df_new["CNation"] = encoder_df["CNation"]
    df_new["CNation_fm21"] = encoder_df["CNation_fm21"]
    labelencoder = LabelEncoder()
    encoder_df = df_new[["CCity", "CCity_fm21"]]
    labelencoder.fit(encoder_df.stack().unique())
    encoder_df['CCity'] = labelencoder.transform(encoder_df['CCity'])
    encoder_df['CCity_fm21'] = labelencoder.transform(encoder_df['CCity_fm21'])
    df_new["CCity"] = encoder_df["CCity"]
    df_new["CCity_fm21"] = encoder_df["CCity_fm21"]
    ############
    labelencoder = LabelEncoder()
    encoder_df = df_new[["Team", "Team_fm21"]]
    labelencoder.fit(encoder_df.stack().unique())
    encoder_df['Team'] = labelencoder.transform(encoder_df['Team'])
    encoder_df['Team_fm21'] = labelencoder.transform(encoder_df['Team_fm21'])
    df_new["Team"] = encoder_df["Team"]
    df_new["Team_fm21"] = encoder_df["Team_fm21"]
    ##################
    labelencoder = LabelEncoder()
    encoder_df = df_new[["CLeague", "CLeague_fm21"]]
    labelencoder.fit(encoder_df.stack().unique())
    encoder_df['CLeague'] = labelencoder.transform(encoder_df['CLeague'])
    encoder_df['CLeague_fm21'] = labelencoder.transform(encoder_df['CLeague_fm21'])
    df_new["CLeague"] = encoder_df["CLeague"]
    df_new["CLeague_fm21"] = encoder_df["CLeague_fm21"]

    # Label Encoder ile yapıldı
    df_new = one_hot_encoder(df_new, ["Foot","Foot_fm21", "Position", "Age_Potential_Seg","Kategori"], drop_first=True)

    ############
    # Scaling
    #############
    rs = RobustScaler()
    num_cols = [i for i in num_cols if i not in "Wages"]
    df_new[num_cols] = rs.fit_transform(df_new[num_cols])
    rs.inverse_transform(df_new[num_cols])
    df_new[num_cols] = rs.inverse_transform(df_new[num_cols])

    # #log1p ---> Sale_Price
    # df_new['Wages'] = np.log1p(df_new["Wages"].values)
    return df_new
def principal_comp_anlys(df_new, pca_plot=False):

    ################################
    # Principal Component Analysis
    ################################
    pca_col = [i for i in df_new.columns if i not in "Wages"]
    # Optimum Bileşen Sayısı
    pca = PCA().fit(df_new[pca_col])
    if pca_plot:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Bileşen Sayısını")
        plt.ylabel("Kümülatif Varyans Oranı")
        plt.show()

    ################################
    # Final PCA'in Oluşturulması
    ################################
    pca = PCA(n_components=11)
    pca_fit = pca.fit_transform(df_new[pca_col])
    final_df = pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11"])
    final_df["Wages"] = df_new["Wages"].reset_index(drop=True)
    return final_df


#########################################
# Model Kurma
#########################################
def models_(X_train, X_test, y_train, y_test,y,log=False):
    models = []
    models.append(('RF', RandomForestRegressor()))
    models.append(('GBM', GradientBoostingRegressor()))
    models.append(("XGBoost", XGBRegressor(objective='reg:squarederror')))
    models.append(("LightGBM", LGBMRegressor()))
    models.append(("CatBoost", CatBoostRegressor(verbose=False)))
    names = []
    rmse = []
    if log:
        #log
        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse.append(np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred))))
            names.append(name)
        tr_split = pd.DataFrame({'Name': names, 'RMSE': rmse})
        tr_split = tr_split.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
        print(tr_split,"\n")
        print(" Mean: ",np.expm1(y).mean(),"\n","Median: ",np.expm1(y).median(),"\n","Std: ",np.expm1(y).std())
    else:
        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            names.append(name)
        tr_split = pd.DataFrame({'Name': names, 'RMSE': rmse})
        tr_split = tr_split.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
        print(tr_split,"\n")
        print(" Mean: ",y.mean(),"\n","Median: ",y.median(),"\n","Std: ",y.std())
        sorted_models = []
        for name in tr_split["Name"]:
            for i, col in enumerate(models):
                if col[0] == name:
                    sorted_models.append(col)

    return "LightGBM",sorted_models
def best_model(X_train, X_test, y_train, y_test,y,plot=False):
    model_name,models = models_(X_train, X_test, y_train, y_test,y)
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
            return df_feature,modelfit,models
def feature_model(final_df,df_feature_model):
    #####################################
    # Feature Importance New Model
    ####################################
    importance_col = df_feature_model[df_feature_model["Value"]!=0]["Feature"].tolist()
    y = final_df["Wages"]
    X = final_df[importance_col]
    X.info()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=17)
    best_model(X_train, X_test, y_train, y_test,plot=True)
def hyperparameter_optimization(modelfit,model_params,X_train,y_train,X_test,y_test,cv=3):
    model_best_grid = GridSearchCV(modelfit, model_params,
                              cv=cv,
                              n_jobs=-1,
                              verbose=True).fit(X_train, y_train)
    model_final = modelfit.set_params(**model_best_grid.best_params_, random_state=17).fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    cv_results = cross_validate(model_final, X_train, y_train, cv=cv, scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])
    print("CV RMSE   : ", -(cv_results['test_neg_root_mean_squared_error'].mean()))
    print("CV MAE    : ", -(cv_results['test_neg_mean_absolute_error'].mean()))
    print("CV R-KARE :", (cv_results['test_r2'].mean()))
    return model_final
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
def catb_overfit(df_new):
    # Catboost ile overfitting önleme
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
    X = df_new.drop(["Wages"], axis=1)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
    train_pool = Pool(X, y)
    eval_pool = Pool(X_valid, y_valid)
    model = CatBoostRegressor(iterations=2500, learning_rate=0.1, eval_metric=RmseMetric())
    modelfit = model.fit(X_train, y_train, eval_set=eval_pool, early_stopping_rounds=10)
    y_pred = modelfit.predict(X_test)
    print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    # Train Rmse
    y_pred_tr = modelfit.predict(X_train)
    print("Train RMSE: ", np.sqrt(mean_squared_error(y_train, y_pred_tr)))
def main():
    df_new,df_clubs,df_fm21,df_clubs_fm21 = load_data()
    df_new,df_name_Cname = data_prep(df_new,df_clubs,df_fm21,df_clubs_fm21)
    final_df = feature_eng(df_new)
    y = final_df["Wages"]
    X = final_df.drop("Wages", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
    df_feature_model, modelfit, models = best_model(X_train, X_test, y_train, y_test, y, plot=True)
    # feature_model(final_df, df_feature_model)

    model_params = {'max_depth': range(-1, 11),
                    "min_child_samples": range(10, 30, 5),
                    'n_estimators': range(100, 1000, 100),
                    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                    "colsample_bytree": [0.9, 0.8, 1]}
    model_final = hyperparameter_optimization(modelfit, model_params, X_train, y_train, X_test, y_test, cv=3)
    voting_clf = voting_regression(models,X_train,y_train)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

if __name__ == "__main__":
    print("İşlem başladı")
    main()
