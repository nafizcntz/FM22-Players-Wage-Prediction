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
df_X_test = pd.read_csv(r"C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\Model_X_test_2.csv")

model = joblib.load(r'C:\Users\Nafiz\Python\Turkcell GY DS Bootcamp\Final Projesi\Turkcell GY DS Bootcamp Projesi\Streamlit\voting_clf_2.pkl')
df_X_test["Wages"]
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
        X = final_df.drop(["Wages", "Name", "Img_Link"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
        return model.predict(X.iloc[index].values.reshape(1, -1))
        y_pred = model.predict(X)
df_test = pd.DataFrame(X_test)
df_2["y_pred"]=y_pred
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)
df_test = df_2.loc[X_test.index]
df_test["CLeague"]
df_test.to_csv("Model_X_test_2.csv", index=False)
len(y_pred)
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
        df_grpby_1 = pd.DataFrame(df_X_test[(df_X_test["CNation"] == str("Argentina"))&(df_X_test["CLeague"] == str("Argentina Superliga Argentina"))].groupby('Team').apply(rmse).reset_index())
        chart_3 = alt.Chart(df_grpby_1).mark_bar().encode(x=alt.X("Team", sort='-y'), y="rmse").interactive()
        st.altair_chart(chart_3)
        df_X_test["Team"]
    else:
        df_grpby_2 = pd.DataFrame(df_X_test[(df_X_test["CNation"] == str(show_me_Nation)) & (df_X_test["CLeague"] == str(show_me_league))].groupby('Team').apply(mae).reset_index())
        chart_4 = alt.Chart(df_grpby_2).mark_bar().encode(x=alt.X("Team", sort='-y'), y="mae").interactive()
        st.altair_chart(chart_4)

























