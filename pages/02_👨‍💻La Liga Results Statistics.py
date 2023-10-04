import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

matchData = st.session_state['matchdata']
st.header("Win, Draw or Loss Statistics of La Liga in Past 7 Years")
fig = go.Figure(data=[go.Pie(labels=matchData.Result, values=matchData.Result_Count, pull=[0, 0, 0.1, 0])])
fig.update_layout(title_text='Results of La Liga (2015-2021)')
st.plotly_chart(fig)


resultsData = pd.read_excel('results.xlsx')

st.header("Win % of Each Team (2015-2021)")
fig = px.bar(resultsData,x='Team',y="Win%",color="Team")
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)

st.header("Loss % of Each Team (2015-2021)")
fig = px.bar(resultsData,x='Team',y="Loss%",color="Team")
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)

st.header("Draw % of Each Team (2015-2021)")
fig = px.bar(resultsData,x='Team',y="Draw%",color="Team")
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)