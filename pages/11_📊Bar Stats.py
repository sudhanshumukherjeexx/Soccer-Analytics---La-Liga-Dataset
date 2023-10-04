import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

matchData = st.session_state['matchdata']

x_axis_val = st.selectbox('Select X-Axis value : [ Team ]',options=matchData.columns)
y_axis_val = st.selectbox('Select Y-Axis value : [ SELECT ANY NUMERIC FEATURE ]',options=matchData.columns)
fig = px.bar(matchData, x=x_axis_val, y=y_axis_val, color="Season", width= 1000, height=600)
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)


