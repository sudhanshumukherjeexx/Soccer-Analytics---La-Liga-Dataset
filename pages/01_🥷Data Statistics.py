import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

matchData = st.session_state['matchdata']


st.header('Data Statistics')
st.write(matchData.describe())

st.header('Data Header')
st.write(matchData.head())

st.header('Correlation in Dataset')
st.write(matchData.corr(numeric_only=True))

st.header('Correlation Heatmap of our Dataset')
fig,ax = plt.subplots(1,1)
sns.set(rc={'figure.figsize':(30,15)})
sns.heatmap(matchData.corr(numeric_only=True),cmap='RdYlGn',annot=True,ax=ax)
st.pyplot(fig)
