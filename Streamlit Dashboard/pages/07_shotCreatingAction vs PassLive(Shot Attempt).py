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
options = st.sidebar.radio('La Liga Teams', options=['Shot Creating Action(All Season)','Atletico Madrid', 'Athletic Club', 'Barcelona', 'Real Betis',
       'Celta Vigo', 'Eibar', 'Espanyol', 'Getafe', 'Granada',
       'La Coruna', 'Las Palmas', 'Levante', 'Malaga', 'Rayo Vallecano',
       'Real Madrid', 'Real Sociedad', 'Sevilla', 'Sporting Gijon',
       'Valencia', 'Villarreal', 'Alaves', 'Leganes', 'Osasuna', 'Girona',
       'Valladolid', 'Huesca', 'Mallorca', 'Cadiz', 'Elche'])


def sca_p(dataframe):
    fig = px.scatter(dataframe,x="shotCreatingAction",y="PassLive(LeadingtoShotAttempt)",color="Season",symbol="Season",hover_name="Opponent",size_max=60)
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig)



TeamData = matchData.groupby('Team')
atl = TeamData.get_group("Atletico Madrid")
atc = TeamData.get_group("Athletic Club")
barcelona = TeamData.get_group("Barcelona")
realBetis = TeamData.get_group("Real Betis")
celtaVigo = TeamData.get_group("Celta Vigo")
eibar = TeamData.get_group("Eibar")
espanyol = TeamData.get_group("Espanyol")
getafe = TeamData.get_group("Getafe")
granada = TeamData.get_group("Granada")
lacoruna = TeamData.get_group("La Coruna")
laspalmas = TeamData.get_group("Las Palmas")
levante = TeamData.get_group("Levante")
malaga = TeamData.get_group("Malaga")
rayo = TeamData.get_group("Rayo Vallecano")
realMadrid = TeamData.get_group("Real Madrid")
realSociedad = TeamData.get_group("Real Sociedad")
sevilla = TeamData.get_group("Sevilla")
sgijon = TeamData.get_group("Sporting Gijon")
valencia = TeamData.get_group("Valencia")
villarReal = TeamData.get_group("Villarreal")
alaves = TeamData.get_group("Alaves")
leganes = TeamData.get_group("Leganes")
osasuna = TeamData.get_group("Osasuna")
girona = TeamData.get_group("Girona")
valladolid = TeamData.get_group("Valladolid")
huesca = TeamData.get_group("Huesca")
mallorca = TeamData.get_group("Mallorca")
cadiz = TeamData.get_group("Cadiz")
elche = TeamData.get_group("Elche")


sca = matchData.groupby(["Season","Team"],as_index=False)["shotCreatingAction"].sum()
sca = sca.sort_values(by=["Season","shotCreatingAction"],ascending=[True,False])


if options == 'Atletico Madrid':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(atl)
elif options == 'Athletic Club':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(atc)
elif options == 'Barcelona':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(barcelona)
elif options == 'Real Betis':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(realBetis)
elif options == 'Celta Vigo':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(celtaVigo)
elif options == 'Eibar':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(eibar)
elif options == 'Espanyol':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(espanyol)
elif options == 'Getafe':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(getafe)
elif options == 'Granada':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(granada)
elif options == 'La Coruna':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(lacoruna)
elif options == 'Las Palmas':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(laspalmas)
elif options == 'Levante':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(levante)
elif options == 'Malaga':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(malaga)
elif options == 'Rayo Vallecano':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(rayo)
elif options == 'Real Madrid':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(realMadrid)
elif options == 'Real Sociedad':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(realSociedad)
elif options == 'Sevilla':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(sevilla)
elif options == 'Sporting Gijon':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(sgijon)
elif options == 'Valencia':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(valencia)
elif options == 'Villarreal':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(villarReal)
elif options == 'Alaves':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(alaves)
elif options == 'Leganes':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(leganes)
elif options == 'Osasuna':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(osasuna)
elif options == 'Girona':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(girona)
elif options == 'Valladolid':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(valladolid)
elif options == 'Huesca':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(huesca)
elif options == 'Mallorca':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(mallorca)
elif options == 'Cadiz':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(cadiz)
elif options == 'Elche':
    st.markdown("#### Shot Creating Action vs Pass Live (Leading to Shot Attempt)")
    sca_p(elche)
elif options == 'Shot Creating Action(All Season)':
    st.markdown("#### Shot Creating Actions (All Season)")
    fig = px.scatter(sca, x="Team", y="shotCreatingAction", color="Season",
                 hover_data=['Team','shotCreatingAction','Season'])
    fig.update_layout(plot_bgcolor='azure')
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig)