import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Match Data (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')
st.text(' This app is designed to work with the Match Dataset in Github repsitory.') 
st.text('Please Download the dataset from Github Respository') 
st.write("Github Repository [Link](https://github.com/sudhanshumukherjeexx/Soccer-Analytics---La-Liga-Dataset)")
st.image("laliga.png",output_format="auto")

st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader("Upload your file here")

if uploaded_file:
    #st.header('Data Statistics')
    matchData = pd.read_excel(uploaded_file)
    matchData['xG'] = matchData['xG'].fillna(matchData.groupby('Team')['xG'].transform('mean'))
    matchData['xGA'] = matchData['xGA'].fillna(matchData.groupby('Team')['xGA'].transform('mean'))
    matchData['shotCreatingAction'] = matchData['shotCreatingAction'].fillna(matchData.groupby('Team')['shotCreatingAction'].transform('mean'))
    matchData['PassLive(LeadingtoShotAttempt)'] = matchData['PassLive(LeadingtoShotAttempt)'].fillna(matchData.groupby('Team')['PassLive(LeadingtoShotAttempt)'].transform('mean'))
    matchData['PassDead(LeadingtoShotAttempt)'] = matchData['PassDead(LeadingtoShotAttempt)'].fillna(matchData.groupby('Team')['PassDead(LeadingtoShotAttempt)'].transform('mean'))
    matchData['dribblesLeadingToShot'] = matchData['dribblesLeadingToShot'].fillna(matchData.groupby('Team')['dribblesLeadingToShot'].transform('mean'))
    matchData['goalCreatingAction'] = matchData['goalCreatingAction'].fillna(matchData.groupby('Team')['goalCreatingAction'].transform('mean'))
    matchData['PassLive(LeadingtoGoal)'] = matchData['PassLive(LeadingtoGoal)'].fillna(matchData.groupby('Team')['PassLive(LeadingtoGoal)'].transform('mean'))
    matchData['PassDead(LeadingtoGoal)'] = matchData['PassDead(LeadingtoGoal)'].fillna(matchData.groupby('Team')['PassDead(LeadingtoGoal)'].transform('mean'))
    matchData['dribblesLeadingToGoals'] = matchData['dribblesLeadingToGoals'].fillna(matchData.groupby('Team')['dribblesLeadingToGoals'].transform('mean'))
    matchData['Shots_on_target%'] = matchData['Shots_on_target%'].fillna(matchData.groupby('Team')['Shots_on_target%'].transform('mean'))
    matchData['Goals/Shot'] = matchData['Goals/Shot'].fillna(matchData.groupby('Team')['Goals/Shot'].transform('mean'))
    matchData['Goals/ShotsonTarget'] = matchData['Goals/ShotsonTarget'].fillna(matchData.groupby('Team')['Goals/ShotsonTarget'].transform('mean'))
    matchData['Distance_from_goal_scored'] = matchData['Distance_from_goal_scored'].fillna(matchData.groupby('Team')['Distance_from_goal_scored'].transform('mean'))
    matchData['Freekick'] = matchData['Freekick'].fillna(matchData.groupby('Team')['Freekick'].transform('mean'))
    matchData['Result_Count'] = matchData['Result'].map({'W': 1, 'L': 2, 'D':3})
    st.session_state['matchdata'] = matchData
    