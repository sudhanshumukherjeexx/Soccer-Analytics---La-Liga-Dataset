import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title('La Liga Matches (2015 - 2021)')
st.text('This is a web app to explore La Liga Matches Data in Last 7 Seasons of La Liga')

matchData = st.session_state['matchdata']

st.header('Individual Team : Win, Loss and Draws')

#----------------------------------------------------------------------------------------
team_list = list(matchData['Team'].unique())
team_win_dic={}
for team in team_list:
    win_count = 0
    for i in range(len(matchData)):
        if matchData['Result'][i] == 'W' and matchData['Team'][i]== team:
            win_count += 1   
    team_win_dic[team] = win_count
teams_win = sorted(team_win_dic.items(), key=lambda team:team[1], reverse=True)
teams_win = pd.DataFrame(teams_win, columns=["Team", "Number of Wins"])
#----------------------------------------------------------------------------------------
team_list = list(matchData['Team'].unique())
team_loss_dic={}
for team in team_list:
    loss_count = 0
    for i in range(len(matchData)):
        if matchData['Result'][i] == 'L' and matchData['Team'][i]== team:
            loss_count += 1
    team_loss_dic[team] = loss_count
teams_loss = sorted(team_loss_dic.items(), key=lambda team:team[1], reverse=True)
teams_loss = pd.DataFrame(teams_loss, columns=["Team", "Number of Loss"])
#----------------------------------------------------------------------------------------
team_list = list(matchData['Team'].unique())
team_draw_dic={}
for team in team_list:
    draw_count = 0
    for i in range(len(matchData)):
        if matchData['Result'][i] == 'D' and matchData['Team'][i]== team:
            draw_count += 1
    team_draw_dic[team] = draw_count
teams_draw = sorted(team_draw_dic.items(), key=lambda team:team[1], reverse=True)
teams_draw = pd.DataFrame(teams_draw, columns=["Team", "Number of Draw"])
#----------------------------------------------------------------------------------------
teamData = pd.merge(pd.merge(teams_win,teams_loss,on='Team'),teams_draw,on='Team')


fig = go.Figure()
fig.add_trace(go.Scatter(x=teams_win['Team'], y=teams_win['Number of Wins'], mode = 'lines+markers'))
fig.update_layout(title="Matches won by Individual Teams(2015-2021)")
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)


fig = go.Figure()
fig.add_trace(go.Scatter(x=teams_loss['Team'], y=teams_loss['Number of Loss'], mode = 'lines+markers'))
fig.update_layout(title="Matches Lost by Individual Teams(2015-2021)")
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)


fig = go.Figure()
fig.add_trace(go.Scatter(x=teams_draw['Team'], y=teams_draw['Number of Draw'], mode = 'lines+markers'))
fig.update_layout(title="Matches Drawn for Individual Teams(2015-2021)")
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)