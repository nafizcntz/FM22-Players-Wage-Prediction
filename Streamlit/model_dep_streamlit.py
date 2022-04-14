import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib

st.set_page_config(layout="wide")

####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('BuLiAn - Bundesliga Analyzer')
with row0_2:
    st.text("")
    st.subheader('Github [Github](https://github.com/nafizcntz/FM22-Players-Wage-Prediction)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown(
        "Hello there! Have you ever spent your weekend watching the German Bundesliga and had your friends complain about how 'players definitely used to run more' ? However, you did not want to start an argument because you did not have any stats at hand? Well, this interactive application containing Bundesliga data from season 2013/2014 to season 2019/2020 allows you to discover just that! If you're on a mobile device, I would recommend switching over to landscape for viewing ease.")
    st.markdown("You can find the source code in the [BuLiAn GitHub Repository](https://github.com/tdenzl/BuLiAn)")


df = pd.read_csv(r"C:\Users\ezelb\OneDrive\Belgeler\GitHub\FM22-Players-Wage-Prediction\Model Deployment\Model_Deployment.csv")
### SEE DATA ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader("Currently selected data:")

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
with row2_1:
    unique_players_in_df = df.Name.nunique()
    str_players = "🏟️ " + str(unique_players_in_df) + " Players"
    st.markdown(str_players)
with row2_2:
    unique_teams_in_df = len(np.unique(df.Team).tolist())
    t = " Teams"
    if(unique_teams_in_df==1):
        t = " Team"
    str_teams = "🏃‍♂️ " + str(unique_teams_in_df) + t
    st.markdown(str_teams)
with row2_3:
    total_league_in_df = len(np.unique(df.CLeague).tolist())
    str_league = "🥅 " + str(total_league_in_df) + " League"
    st.markdown(str_league)
# with row2_4:
#     total_shots_in_df = df_data_filtered['shots_on_goal'].sum()
#     str_shots = "👟⚽ " + str(total_shots_in_df) + " Shots"
#     st.markdown(str_shots)

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.markdown("")
    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=df.reset_index(drop=True))
st.text('')

############################
# Model deploymen
###########################

### TEAM SELECTION ###
#unique_teams = get_unique_teams(df_data_filtered_matchday)
st.sidebar.header("Player Wage Model Prediction")
all_nation_selected = st.sidebar.selectbox("Choose the Player's Nation of League ",
                                           [""]+df.CNation.value_counts().index.tolist(),
                                           format_func=lambda x: "" if x == "" else x)

all_league_selected = st.sidebar.selectbox("Choose the player's League",
                                           [""]+df[df.CNation == all_nation_selected]["CLeague"].value_counts().index.tolist(),
                                           format_func=lambda x: "" if x == "" else x)

all_teams_selected = st.sidebar.selectbox("Choose the player's Team",
                                          [""]+df[df.CLeague == all_league_selected]["Team"].value_counts().index.tolist(),
                                          format_func=lambda x: "" if x == "" else x)

all_player_selected = st.sidebar.selectbox("Choose the player's Name",
                                           [""]+df[df.Team == all_teams_selected]["Name"].value_counts().index.tolist(),
                                           format_func=lambda x: "" if x == "" else x)

def fm22_prediction(df,name):
    if name == "":
        return ""
    else:
        index = df[df["Name"] == str(name)].index.tolist()[0]
        final_df = pd.read_csv(r"C:\Users\ezelb\OneDrive\Belgeler\GitHub\FM22-Players-Wage-Prediction\Model Deployment\Model_Deployment.csv")
        model = joblib.load(r'C:\Users\ezelb\OneDrive\Belgeler\GitHub\FM22-Players-Wage-Prediction\Model Deployment\voting_clf_2.pkl')
        y = final_df["Wages"]
        X = final_df.drop(["Wages","Name"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=17)
        return model.predict(X.iloc[index].values.reshape(1, -1))


wage = fm22_prediction(df,all_player_selected)
try:
    st.sidebar.write("Prediction Wage €", wage[0], "pw")
    st.sidebar.write("Real Wage €", df[df["Name"]==all_player_selected]["Wages"].tolist()[0], "pw")
except:
    st.sidebar.write("Choose a Player for Prediction")



################
### ANALYSIS ###
################

### DATA EXPLORER ###
row12_spacer1, row12_1, row12_spacer2 = st.columns((.2, 7.1, .2))
with row12_1:
    st.subheader('Player Information')
    st.markdown('Show the player information...')

row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3, row13_3, row13_spacer4, row13_4, row13_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2, 2.3, .2))
with row13_1:
    show_me_Nation = st.selectbox("Choose the Player's Nation of League ",
                                 [""]+df.CNation.value_counts().index.tolist(),
                                 format_func=lambda x: "" if x == "" else x,
                                key="playernation")

with row13_2:
    show_me_league = st.selectbox("Choose the player's League",
                                 [""]+df[df.CNation == show_me_Nation]["CLeague"].value_counts().index.tolist(),
                                 format_func=lambda x: "" if x == "" else x,
                                key="playerleague")

with row13_3:
    show_me_team = st.selectbox("Choose the player's Team",
                                [""]+df[df.CLeague == show_me_league]["Team"].value_counts().index.tolist(),
                                format_func=lambda x: "" if x == "" else x,
                                key="playerteam")

with row13_4:
    show_me_player = st.selectbox("Choose the player's Name",
                                [""]+df[df.Team == show_me_team]["Name"].value_counts().index.tolist(),
                                format_func=lambda x: "" if x == "" else x,
                                  key="playername")


row15_spacer1, row15_1, row15_2, row15_3, row15_4, row15_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
with row15_1:
    st.subheader("Player")
with row15_2:
    st.subheader("Technical")
with row15_3:
    st.subheader("Mental")
with row15_4:
    st.subheader("Physical")
"""
row17_spacer1, row17_1, row17_spacer2 = st.columns((.2, 7.1, .2))
with row17_1:
       st.warning('Unfortunately this analysis is only available if all teams are included')
"""

row16_spacer1, row16_1, row16_2, row16_3, row16_4, row16_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
with row16_1:
    col_player_info = df.columns[:10]
    for i in df[]:
        st.markdown("'"+df[i] +"'")

with row16_2:
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['shots_on_goal']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['distance']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(df_match_result.iloc[0]['total_passes']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎‎"+str(df_match_result.iloc[0]['possession']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['fouls']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['offside']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[0]['corners']))
with row16_4:
    st.markdown(" "+str(df_match_result.iloc[1]['shots_on_goal']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['distance']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(df_match_result.iloc[1]['total_passes']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(df_match_result.iloc[1]['possession']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['fouls']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['offside']))
    st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(df_match_result.iloc[1]['corners']))
    


