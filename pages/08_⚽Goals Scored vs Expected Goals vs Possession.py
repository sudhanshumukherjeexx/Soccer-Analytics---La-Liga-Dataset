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
options = st.sidebar.radio('La Liga Teams', options=['Atletico Madrid', 'Athletic Club', 'Barcelona', 'Real Betis',
       'Celta Vigo', 'Eibar', 'Espanyol', 'Getafe', 'Granada',
       'La Coruna', 'Las Palmas', 'Levante', 'Malaga', 'Rayo Vallecano',
       'Real Madrid', 'Real Sociedad', 'Sevilla', 'Sporting Gijon',
       'Valencia', 'Villarreal', 'Alaves', 'Leganes', 'Osasuna', 'Girona',
       'Valladolid', 'Huesca', 'Mallorca', 'Cadiz', 'Elche'])


def sca_p(dataframe):
    fig = px.scatter_3d(dataframe, x='Poss', y="GF",z="xG",color="Season",labels={"Poss":"Possession","GF":"Goals Scored","xG":"Expected Goals"})
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



if options == 'Atletico Madrid':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(atl)
elif options == 'Athletic Club':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(atc)
elif options == 'Barcelona':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(barcelona)
elif options == 'Real Betis':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(realBetis)
elif options == 'Celta Vigo':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(celtaVigo)
elif options == 'Eibar':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(eibar)
elif options == 'Espanyol':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(espanyol)
elif options == 'Getafe':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(getafe)
elif options == 'Granada':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(granada)
elif options == 'La Coruna':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(lacoruna)
elif options == 'Las Palmas':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(laspalmas)
elif options == 'Levante':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(levante)
elif options == 'Malaga':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(malaga)
elif options == 'Rayo Vallecano':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(rayo)
elif options == 'Real Madrid':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(realMadrid)
elif options == 'Real Sociedad':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(realSociedad)
elif options == 'Sevilla':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(sevilla)
elif options == 'Sporting Gijon':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(sgijon)
elif options == 'Valencia':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(valencia)
elif options == 'Villarreal':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(villarReal)
elif options == 'Alaves':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(alaves)
elif options == 'Leganes':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(leganes)
elif options == 'Osasuna':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(osasuna)
elif options == 'Girona':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(girona)
elif options == 'Valladolid':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(valladolid)
elif options == 'Huesca':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(huesca)
elif options == 'Mallorca':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(mallorca)
elif options == 'Cadiz':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(cadiz)
elif options == 'Elche':
    st.markdown("#### Goals Scored vs Expected Goals vs Possession")
    sca_p(elche)