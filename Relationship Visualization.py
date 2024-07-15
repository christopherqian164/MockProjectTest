import pandas as pd
import json
import streamlit as st
import plotly.express as px
from pandas.api.types import CategoricalDtype
import numpy as np


st.set_page_config(
    page_title="Relationship Visualization",
    page_icon="👋",
)

var_dict = json.load(open('var_categories_default.json'))

st.header('Survey Response Relationship Visualization')
st.markdown('Select two variables from the BRFSS dataset to visualize their relationship.')

if 'stage' not in st.session_state:
   st.session_state['stage'] = 'start'

option1 = st.selectbox('Select the first variable', var_dict.keys())
l = list(var_dict.keys())
l.remove(option1)
option2 = st.selectbox('Select the second variable', l)

def create_crosstab(df, x_name, y_name):
    x  = df[x_name]
    y  = df[y_name]
    
    if x.dtype.name == 'category' and y.dtype.name == 'category':
        tab = pd.crosstab(x,y)
        prop = tab.div(tab.sum(axis = 1), axis = 0)
        fig = px.bar(prop.reset_index(), x=x_name, y = list(prop.columns), template = 'plotly_dark', title = 'Proportion of responses of ' + y_name + ' for given response to ' + x_name)
        fig.update_xaxes(tickangle=90)
        fig.update_layout(xaxis=dict(
          rangeslider=dict(visible=True)
          )
        )
    elif (x.dtype.name == 'category' and y.dtype.name == 'float32'):
        fig = px.box(df, x = x_name, y = y_name, template='plotly_dark', color = x_name, title = 'Boxplot of responses to ' + y_name + ' for given response to ' + x_name)
        fig.update_traces(boxmean=True, showlegend=False)

    elif x.dtype.name == 'float32' and y.dtype.name == 'category':
        fig = px.box(df, x = x_name, y = y_name, color = y_name, template = 'plotly_dark', title = 'Boxplot of responses of ' + x_name + ' for given response to ' + y_name)
        fig.update_traces(boxmean=True, showlegend=False)

    elif x.dtype.name == 'float32' and y.dtype.name == 'float32':
       fig = px.scatter(df, x = x_name, y = y_name, template = 'plotly_dark', title = 'Scatterplot of ' + x_name + ' vs. ' + y_name)
    return fig

if st.button('Get Prediction'):
  if st.session_state['stage'] == 'start':
    with st.spinner('Downloading data...'):
      df = a = pd.read_sas('https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip', compression = 'zip', format = 'xport')
    with st.spinner('Preprocessing data...'):
      for c in var_dict.keys():
        df[c] = df[c].astype('string')
        var_type = var_dict[c]['var_type']
        var_map = var_dict[c]['var_map']
        df[c] = df[c].replace(var_map)
        df[c] = df[c].replace(
            {'Don\'t know': pd.NA, 'Refused': pd.NA}
        )
        if var_type == 'category':
          df[c] = df[c].astype(
            CategoricalDtype(list(dict.fromkeys(var_map.values())), ordered = True)
          )
        else:
          df[c] = df[c].astype('float32') 

      #Convert sleep time to categorical variable
      df['SLEPTIM1'] = pd.cut(df['SLEPTIM1'], 
                              bins = [0,6,9,24], 
                              labels = ['6 hours or less', '7-9 hours', '10 hours or more'])
      var_dict['SLEPTIM1']['var_type'] = 'category'
      var_dict['SLEPTIM1']['reference_level'] = "7-9 Hours"
      var_dict['SLEPTIM1']['var_map'] = {
          '0': '6 hours or less',
          '1': '7-9 Hours',
          '2': '10 hours or more'
      }

      #convert heights and weights to imperial units
      metric_weights = 2.20462262185 * (
          df.loc[df['WEIGHT2'] > 9000, 'WEIGHT2'] - 9000
          )
      df.loc[df['WEIGHT2'] > 9000, 'WEIGHT2'] = metric_weights

      metric_heights = (3.280839895/100) * (
          df.loc[df['HEIGHT3'] > 9000, 'HEIGHT3'] - 9000)
      df.loc[df['HEIGHT3'] > 9000, 'HEIGHT3'] = metric_heights
      imperial_heights = df.loc[df['HEIGHT3'] <= 9000, 'HEIGHT3']
      imperial_heights = (
          imperial_heights.astype('string').str[:1].astype('float32') + 
          imperial_heights.astype('string').str[1:3].astype('float32')/12)
      df.loc[df['HEIGHT3'] <= 9000, 'HEIGHT3'] = imperial_heights
      df['HEIGHT3'] *= 12

      #replace height and weight with BMI
      df['BMI'] = 702.94925 * df['WEIGHT2'] / (df['HEIGHT3'] ** 2)
      var_dict['BMI'] = {
          'var_type': 'float32',
          'reference_level': '22'
      }

      df.loc[df['CHILDREN'] > 80, 'CHILDREN'] -= 80

      alc_per_week = df.loc[df['ALCDAY4'] < 200, 'ALCDAY4'] - 100
      alc_per_month = df.loc[df['ALCDAY4'] >= 200, 'ALCDAY4'] - 200
      df.loc[df['ALCDAY4'] < 200, 'ALCDAY4'] = alc_per_week * 4
      df.loc[df['ALCDAY4'] >= 200, 'ALCDAY4'] = alc_per_month
      df.loc[df['ALCDAY4'] > 30, 'ALCDAY4'] = 30
      st.session_state['df'] = df
  fig = create_crosstab(st.session_state['df'], option1, option2)
  event = st.plotly_chart(fig)
  st.session_state['stage'] = 'button_pressed'
