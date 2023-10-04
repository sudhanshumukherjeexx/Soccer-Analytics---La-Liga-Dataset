import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

matchData = st.session_state['matchdata']


x_axis = st.selectbox('Select X-Axis value : [ SELECT ANY NUMERIC FEATURE ]',options=matchData.columns)
y_axis = st.selectbox('Select Y-Axis value : [ SELECT ANY NUMERIC FEATURE ]',options=matchData.columns)
fig = px.scatter(matchData, x=x_axis, y=y_axis, color="Team",hover_data=['Season'])
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)