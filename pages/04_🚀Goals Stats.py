import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

matchData = st.session_state['matchdata']

st.header("Goals Stats : Goals Scored and Goals Conceded")
st.markdown( "#### This section will help you visualize the goals stats. You can view Goals Scored by Individual Teams and also Goals Conceded by Individual Teams.")
st.markdown("**Note:** By Default this module was designed to visualize Goals Stats,For better results please select the prompted X_AXIS_VALUE and Y_AXIS_VALUE. You can play around it by passing different features available in the dropdown menu,However this might throw an error on some features selected.")

x_axis_val = st.selectbox('Select X-Axis value : Team',options=matchData.columns)
y_axis_val = st.selectbox('Select Y-Axis value : GF = Goals Scored, GA = Goals Conceded',options=matchData.columns)
fig = px.bar(matchData, x=x_axis_val, y=y_axis_val, color="Season", width= 1000, height=600)
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)

