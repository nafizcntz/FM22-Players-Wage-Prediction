import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
import joblib
import streamlit.components.v1 as components

st.set_page_config(layout="wide")


df = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\Model_deployment.csv")
df_2 = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\Model_deployment_2.csv")
df_X_test = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\Model_X_test.csv")

model = joblib.load(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\voting_clf_2.pkl')

############################
# Model deployment
###########################

def fm22_prediction(df, name):
    if name == "":
        return ""
    else:
        index = df[df["Name"] == str(name)].index.tolist()[0]
        final_df = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\Model_deployment.csv")
        model = joblib.load(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\voting_clf_2.pkl")
        y = final_df["Wages"]
        X = final_df.drop(["Wages","Name", "Img_Link"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
        return model.predict(X.iloc[index].values.reshape(1, -1))
        model.predict(X)
wage = fm22_prediction(df, str(all_player_selected))

try:
    rw = df[df["Name"] == all_player_selected]["Wages"].tolist()[0]
    pw = wage[0]
    st.sidebar.write("Prediction Wage **€ {:.2f}** pw".format(pw))
    st.sidebar.write("Real Wage **€ {:.2f}** pw".format(rw))
    ps = abs(rw-pw) / rw
    st.sidebar.write("Percentage of Deviation in Prediction: % {:.2f}".format(ps))
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
    show_me_Nation = st.selectbox("Nation of the Player's League",
                                df_2.CNation.value_counts().index.tolist(),
                                key="playernation")

with row13_2:
    show_me_league = st.selectbox("Player's League",
                                 df_2[df_2.CNation == show_me_Nation]["CLeague"].value_counts().index.tolist(),
                                 key="playerleague")

with row13_3:
    show_me_team = st.selectbox("Player's Team",
                                df_2[df_2.CLeague == show_me_league]["Team"].value_counts().index.tolist(),
                                key="playerteam")

with row13_4:
    show_me_player = st.selectbox("Player's Name",
                                  df_2[df_2.Team == show_me_team]["Name"].value_counts().index.tolist(),
                                  key="playername")


#row15_spacer1, row15_1, row15_2, row15_3, row15_4, row15_spacer2  = st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
row15_spacer1, row15_1, row15_spacer2, row15_2, row15_spacer3, row15_3, row15_spacer4, row15_4, row15_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2, 2.3, .2))
with row15_1:
    st.subheader("Player")
with row15_2:
    st.subheader("Technical")
with row15_3:
    st.subheader("Mental")
with row15_4:
    st.subheader("Physical")

row16_spacer1, row16_1, row16_spacer2, row16_2, row16_spacer3, row16_3, row16_spacer4, row16_4, row16_spacer4 = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2, 2.3, .2))
#row16_spacer1, row16_1, row16_2, row16_3, row16_4, row16_spacer2= st.columns((0.5, 1.5, 1.5, 1, 2, 0.5))
with row16_1:
    col_player_info = df_2.columns[:10]
    k=0
    for i in df_2[col_player_info]:
        if k==0:
            st.image(df_2[df_2["Name"] == str(show_me_player)].loc[:,"Img_Link"].tolist()[0],width=125)

        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_info][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')
        k=k+1
with row16_2:
    col_player_tech = df_2.columns[10:24]
    for i in df_2[col_player_tech]:
        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_tech][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')

with row16_3:
    col_player_mental = df_2.columns[24:38]
    for i in df_2[col_player_mental]:
        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_mental][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')

with row16_4:
    col_player_physical = df_2.columns[38:46]
    for i in df_2[col_player_physical]:
        st.markdown('**' + i + '**: ' + '' + str(df_2[col_player_physical][df_2["Name"] == str(show_me_player)].loc[:, i].tolist()[0]) + '')

st.text("")
st.text("")
st.subheader("Mapping Players by Team")
st.write("The Cities of the Players' Team visualized on the World Map")
html = open(r"/app/fm22-players-wage-prediction/Streamlit/Image_Map.html",'r',encoding='utf-8')
source = html.read()
components.html(source,width=1420,height=500)

st.text("")
st.subheader("Map Visualizations")
st.write("Visualizations of football players by country, team, potential and wages on the world map.")
row17_1, row17_2, row17_3, row17_4 = st.columns(4)
a = True
with row17_1:
    button_1 = st.button("Heatmap of the Player's Team Location")
with row17_2:
    button_2 = st.button("Average Wage of Football Players by Country")
    if button_2:
        a = False
with row17_3:
    button_3 = st.button("Potential Average According to the Countries of the Players' Clubs ")
    if button_3:
        a=False
with row17_4:
    button_4 = st.button("Potential Average According to Nationalities of the Players")
    if button_4:
        a = False
if a:
    html = open(
        r"/app/fm22-players-wage-prediction/Streamlit/Nation_Heatmap.html", 'r',
        encoding='utf-8')
    source = html.read()
    components.html(source, width=1420, height=500)

if button_2:
    html = open(
        r"/app/fm22-players-wage-prediction/Streamlit/Maaş_ortalamasına_göre_club_ülke_dağılımı.html", 'r',
        encoding='utf-8')
    source = html.read()
    components.html(source, width=1420, height=500)
    a= False
if button_3:
    html = open(
        r"/app/fm22-players-wage-prediction/Streamlit/Potensiyel_ortalamasına_göre_club_ülke_dağılımı.html", 'r',
        encoding='utf-8')
    source = html.read()
    components.html(source, width=1420, height=500)
    a = False
if button_4:
    html = open(
        r"/app/fm22-players-wage-prediction/Streamlit/Potensiyel_ortalamasına_göre_ülke_dağılımı.html", 'r',
        encoding='utf-8')
    source = html.read()
    components.html(source, width=1420, height=500)
    a = False
#4C78A9
#####
st.text('')
st.header("Visualization of Data Analysis")
### Count plot ###
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Distribution of Data by Selected Feature')
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row5_1:
    plot_x_per_team_type = st.selectbox("Which feature do you want to analyze?",["Team","Nation","CLeague","Wages","Ability","Potential","Age"], key= 'measure_team')
with row5_2:
    import altair as alt
    import streamlit as st
    if  df_2[plot_x_per_team_type].dtype== "O":
        chart = alt.Chart(df_2).mark_bar().encode(x=alt.X(plot_x_per_team_type,sort='-y'), y=alt.Y('count()')).interactive()
        st.altair_chart(chart)
    else:
        chart = alt.Chart(df_2).mark_bar().encode(x=plot_x_per_team_type, y='count()').interactive()
        st.altair_chart(chart)

#########
# Nümerik - mümerik
#########
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Distributions of Numerical Features')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    #st.markdown('Investigate a variety of stats for each team. Which team scores the most goals per game? How does your team compare in terms of distance ran per game?')
    plot_x_selection_1 = st.selectbox("Which feature do you want to analyze?",["Wages","Sell_Value","Ability","Potential","Age"], key= 'corr_1')
    plot_x_selection_2 = st.selectbox("Which feature do you want to analyze?",["Potential","Wages","Sell_Value","Ability","Age"], key= 'corr_2')
with row7_2:
    chart = alt.Chart(df_2).mark_circle().encode(x=plot_x_selection_2,y=plot_x_selection_1).interactive()
    st.altair_chart(chart)

######################
# Kategorik - Nümerik
######################

row8_spacer1, row8_1, row8_spacer2 = st.columns((.2, 7.1, .2))
with row8_1:
    st.subheader('Averages of Numerical Feature Group by Categorical Feature')
row9_spacer1, row9_1, row9_spacer2, row9_2, row9_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row9_1:
    #st.markdown('Investigate a variety of stats for each team. Which team scores the most goals per game? How does your team compare in terms of distance ran per game?')
    plot_x_selection_1 = st.selectbox("Which categorical feature do you want to analyze?",["Nation","CCity","Team","Position","Foot"], key= 'num_1')
    plot_x_selection_2 = st.selectbox("Which numerical feature do you want to analyze?",["Potential","Wages","Sell_Value","Ability","Age"], key= 'num_2')
with row9_2:
    chart = alt.Chart(df_2).mark_bar().encode(x=alt.X(plot_x_selection_1,sort='-y'),y="mean("+plot_x_selection_2+")").properties(height=400).interactive()
    st.altair_chart(chart)


######################
# Model Visualizations
######################

row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    st.header('Model Visualizations')
    st.markdown('Data restricted according to the selected country and league;')
    st.markdown('1 - Visualization based on model prediction and actual values')
    st.markdown('2 - Visualization of the model according to the selected metric')
row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3, row11_3  = st.columns((.2, 1.5, .2, 1.5, .2, 1.5))
with row11_1:
    show_me_Nation = st.selectbox("Nation of the Player's League",
                                  df_2.CNation.value_counts().index.tolist(),
                                  key="model_nation")
with row11_2:
    show_me_league = st.selectbox("Nation of the Player's League",
                                 df_2[df_2.CNation == show_me_Nation]["CLeague"].value_counts().index.tolist(),
                                 key="model_league")
with row11_3:
    show_me_metric = st.selectbox("Metrics ",
                                  ["MAE","RMSE"],
                                  key="model_metrics")
row12_spacer1, row12_1, row12_spacer2, row12_2, row12_spacer3 = st.columns((.2, 2.3, 0.8, 4, .4))
with row12_1:
    chart_1 = alt.Chart(df_X_test[df_X_test["CLeague"]==show_me_league]).mark_circle().encode(x="Wages",y="y_pred").interactive()
    chart_2 = alt.Chart(df_X_test[df_X_test["CLeague"]==show_me_league]).mark_line(color="red").encode(x="Wages", y="Wages").interactive()
    st.altair_chart(chart_1+chart_2)
with row12_2:
    def rmse(g):
        rmse = np.sqrt(mean_squared_error(g['Wages'], g['y_pred']))
        return pd.Series(dict(rmse=rmse))
    def mae(g):
        mae = mean_absolute_error(g['Wages'], g['y_pred'])
        return pd.Series(dict(mae=mae))
    if show_me_metric=="RMSE":
        df_grpby_1 = pd.DataFrame(df_X_test[(df_X_test["CNation"] == str(show_me_Nation))&(df_X_test["CLeague"] == str(show_me_league))].groupby('Team').apply(rmse).reset_index())
        chart_3 = alt.Chart(df_grpby_1).mark_bar().encode(x=alt.X("Team", sort='-y'), y="rmse").interactive()
        st.altair_chart(chart_3)
    else:
        df_grpby_2 = pd.DataFrame(df_X_test[(df_X_test["CNation"] == str(show_me_Nation)) & (
                    df_X_test["CLeague"] == str(show_me_league))].groupby('Team').apply(mae).reset_index())
        chart_4 = alt.Chart(df_grpby_2).mark_bar().encode(x=alt.X("Team", sort='-y'), y="mae").interactive()
        st.altair_chart(chart_4)




























