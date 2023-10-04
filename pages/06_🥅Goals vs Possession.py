import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

matchData = st.session_state['matchdata']

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

st.sidebar.title('Navigation')
options = st.sidebar.radio('La Liga Season', options=['2015-2016','2016-2017','2017-2018','2018-2019','2019-2020','2020-2021','2021-2022'])

def poss_goals(dataframe):
    fig = px.histogram(dataframe, x="Poss", y="GF", color="Team", marginal="rug",labels={'Poss':'Possession(%)','GF':'Goals'},hover_data=['Team','Opponent','Result'])
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig)


seasonData = matchData.groupby('Season')
season1 = seasonData.get_group("2015-2016")
season2 = seasonData.get_group("2016-2017")
season3 = seasonData.get_group("2017-2018")
season4 = seasonData.get_group("2018-2019")
season5 = seasonData.get_group("2019-2020")
season6 = seasonData.get_group("2020-2021")
season7 = seasonData.get_group("2021-2022")

if options == '2015-2016':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season1)
elif options == '2016-2017':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season2)
elif options == '2017-2018':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season3)
elif options == '2018-2019':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season4)
elif options == '2019-2020':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season5)
elif options == '2020-2021':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season6)
elif options == '2021-2022':
    st.markdown("#### Possession vs Goals Scored")
    poss_goals(season7)