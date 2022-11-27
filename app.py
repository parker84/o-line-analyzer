import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px

# -----------helpers

def est_beta_params(mu, var):
  alpha = ((1 - mu) / var - 1 / mu) * (mu ** 2)
  beta = alpha * (1 / mu - 1)
  return alpha, beta


# --------------read in data
scout_df = pd.read_csv('./data/pffScoutingData.csv')
players_df = pd.read_csv('./data/players.csv')

# -----------data prep
oline_scout_df = scout_df[scout_df['pff_role'] == 'Pass Block']
oline_scout_df = oline_scout_df[[
    'nflId', 'pff_beatenByDefender', 'pff_hitAllowed', 'pff_hurryAllowed', 'pff_sackAllowed'
]]
oline_scout_df['successful_block'] = (
    oline_scout_df['pff_beatenByDefender'] +
    oline_scout_df['pff_hitAllowed'] + 
    oline_scout_df['pff_hurryAllowed'] + 
    oline_scout_df['pff_sackAllowed']
) == 0
oline_scout_df['count_plays'] = 1
oline_performance_df = oline_scout_df.groupby('nflId').sum()
oline_performance_df['percent_successful_blocks'] = (
    oline_performance_df['successful_block'] / 
    oline_performance_df['count_plays']
)
oline_performance_df = oline_performance_df.merge(
    players_df, on='nflId'
)
oline_performance_df = oline_performance_df[
    oline_performance_df['officialPosition'].isin(['G', 'TE', 'C', 'T'])
]

# -----------smoothing (empirical bayes estimation)
oline_performance_df_for_fitting = oline_performance_df[
    oline_performance_df['count_plays'] >= 100
]
alpha0, beta0 = est_beta_params(
    oline_performance_df_for_fitting['percent_successful_blocks'].mean(),
    oline_performance_df_for_fitting['percent_successful_blocks'].std()**2
)

oline_performance_df['smoothed_percent_successful_blocks'] = (
    oline_performance_df['successful_block'] + alpha0
) / (oline_performance_df['count_plays'] + alpha0 + beta0)



#-----------------streamlit setup
nfl_logo_image = Image.open('./assets/nlf_logo.png')
st.set_page_config(layout='wide', page_icon=nfl_logo_image, page_title='NFL O-Line Dashboard')
st.title('NFL O-Line Dashboard')
st.markdown(
    """
    **Purpose**: The purpose of this dashboard is to enable NFL Coaches to compare Offensive Lineman's Performance (with one metric)
    even when one player has experienced way more snaps than another (ex: 10 vs 300)
    """
)
st.sidebar.image(nfl_logo_image, width=150)
positions = st.sidebar.multiselect('Filter by Position', ['Select All'] + oline_performance_df['officialPosition'].value_counts().index.tolist(), ['Select All'])
names = st.sidebar.multiselect('Filter by Name', ['Select All'] + oline_performance_df['displayName'].value_counts().index.tolist(), ['Select All'])
colleges = st.sidebar.multiselect('Filter by College', ['Select All'] + oline_performance_df['collegeName'].value_counts().index.tolist(), ['Select All'])


# ---------------filtering
if 'Select All' not in positions:
    oline_performance_df = oline_performance_df[oline_performance_df['officialPosition'].isin(positions)]
if 'Select All' not in names:
    oline_performance_df = oline_performance_df[oline_performance_df['displayName'].isin(names)]
if 'Select All' not in colleges:
    oline_performance_df = oline_performance_df[oline_performance_df['collegeName'].isin(colleges)]

# --------------cleaning for data viz
oline_performance_df = oline_performance_df.rename(columns={
    'pff_beatenByDefender': 'Count Plays Beaten By Defender',
    'pff_hitAllowed': 'Count QB Hits Allowed',
    'pff_hurryAllowed': 'Count QB Hurrys Allowed',
    'pff_sackAllowed': 'Count QB Sacks Allowed',
    'successful_block': 'Count Successful Blocks',
    'smoothed_percent_successful_blocks': 'EB % of Blocks Successful',
    'percent_successful_blocks': '% of Blocks Successful',
    'count_plays': 'Count Plays',
    'displayName': 'Player Name',
    'collegeName': 'College Name',
    'officialPosition': 'Position',
    'height': 'Height',
    'weight': 'Weight',
    'birthDate': 'Birth Date'
}).sort_values(by='EB % of Blocks Successful', ascending=False)
oline_performance_df = oline_performance_df[[
    'Player Name',
    'Position',
    'EB % of Blocks Successful',
    '% of Blocks Successful',
    'Count Plays',
    'Count Successful Blocks',
    'Count Plays Beaten By Defender',
    'Count QB Hits Allowed',
    'Count QB Hurrys Allowed',
    'Count QB Sacks Allowed',
    'College Name',
    'Height',
    'Weight',
    'Birth Date'
]]
oline_performance_df.index = range(1, oline_performance_df.shape[0]+1)

# --------------making the dashboard
st.markdown('## Offensive Linemen Rankings')
st.markdown(
    """
    #### Metric Definitions
    1. First we use `% of Blocks Successful` = % of blocks the lineman made where they didn't allow a sack, hurry, hit, and were not beaten by the defender.
    2. Then we use a smoothing approach (empirical bayes) that brings the stats for the players with less snaps closer to the overall mean = `EB % of Blocks Successful`

    Now you can see how this produces the rankings below:
    """
)
st.dataframe(oline_performance_df)

st.markdown('### Smoothing Effects')
st.markdown('When players have more plays their `% of Blocks Successful` and `EB % of Blocks Successful` are heavily correlated.')
st.markdown('But for players with fewer plays the `EB % of Blocks Successful` is brought closer to the mean')
fig = px.scatter(
    oline_performance_df[oline_performance_df['% of Blocks Successful'] > 0.6], # we filter out outliers to make it easier to read
    title='Impact of Empirical Bayes Smoothing on `% of Blocks Successful` Estimates by Play Counts',
    x='% of Blocks Successful',
    y='EB % of Blocks Successful', 
    color='Count Plays')
st.plotly_chart(fig)