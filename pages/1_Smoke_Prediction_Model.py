import streamlit as st
import pandas as pd
import statsmodels.api as sm
import json
import plotly.express as px
import numpy as np
import patsy

def preprocess(df, var_dict):

  covariates = [
  '_STATE',
  'GENHLTH','PHYSHLTH','MENTHLTH',
  'PRIMINSR','PERSDOC3','MEDCOST1','CHECKUP1',
  'EXERANY2','SLEPTIM1',
  'LASTDEN4','RMVTETH4',
  'MARITAL','EDUCA','RENTHOM1','VETERAN3','EMPLOY1','CHILDREN',
  'INCOME3','WEIGHT2','HEIGHT3',
  'ALCDAY4',
  'FLUSHOT7','PNEUVAC4','TETANUS1',
  'HIVTST7','HIVRISK5',
  'COVIDPOS'
  ]
  response = 'SMOKDAY2'

  dfX = df[covariates].copy()

  #Map numeric values to corresponding answers
  for c in covariates:
    dfX[c] = dfX[c].astype('string')
    var_type = var_dict[c]['var_type']
    var_map = var_dict[c]['var_map']
    dfX[c] = dfX[c].replace(var_map)
    dfX[c] = dfX[c].replace(
      {'Don\'t know': pd.NA, 'Refused': pd.NA}
      ) 
    if var_type == 'category':
      categories = list(dict.fromkeys(var_map.values()))
      if 'Don\'t know' in categories:
        categories.remove('Don\'t know')
      if 'Refused' in categories:
        categories.remove('Refused') 
      var_dict[c]['categories'] = categories
    if var_type == 'float32':
      dfX[c] = dfX[c].astype(var_type)                

  #convert states into regions
  reduce_dict = json.load(open('reduce_categories.json'))
  
  for c in covariates:
    if c in reduce_dict.keys():
      reduce_map = reduce_dict[c]
      dfX[c] = dfX[c].replace(reduce_map)
      for e in reduce_map.keys():
        var_dict[c]['categories'].remove(e)
      for e in list(dict.fromkeys(reduce_map.values())):
        var_dict[c]['categories'].append(e)
      
  var_dict['_STATE']['reference_level'] = 'South Atlantic'
  var_dict['TETANUS1']['reference_level'] = 'Yes'

  #Convert sleep time to categorical variable
  dfX['SLEPTIM1'] = pd.cut(dfX['SLEPTIM1'], 
                          bins = [0,6,9,24], 
                          labels = ['6 hours or less', 
                                    '7-9 hours', 
                                    '10 hours or more'])
  var_dict['SLEPTIM1']['var_type'] = 'category'
  var_dict['SLEPTIM1']['reference_level'] = "7-9 hours"
  var_dict['SLEPTIM1']['categories'] = ['6 hours or less',
                                          '7-9 hours',
                                          '10 hours or more']

  #convert heights and weights to imperial units
  # metric_weights = 2.20462262185 * (
  #   dfX.loc[dfX['WEIGHT2'] > 9000, 'WEIGHT2'] - 9000
  #   )
  # dfX.loc[dfX['WEIGHT2'] > 9000, 'WEIGHT2'] = metric_weights

  # metric_heights = (3.280839895/100) * (
  #   dfX.loc[dfX['HEIGHT3'] > 9000, 'HEIGHT3'] - 9000)
  # dfX.loc[dfX['HEIGHT3'] > 9000, 'HEIGHT3'] = metric_heights
  # imperial_heights = dfX.loc[dfX['HEIGHT3'] <= 9000, 'HEIGHT3']
  # imperial_heights = (
  #   imperial_heights.astype('string').str[:1].astype('float32') + 
  #   imperial_heights.astype('string').str[1:3].astype('float32')/12)
  # dfX.loc[dfX['HEIGHT3'] <= 9000, 'HEIGHT3'] = imperial_heights
  # dfX['HEIGHT3'] *= 12

  #replace height and weight with BMI
  dfX['BMI'] = 702.94925 * dfX['WEIGHT2'] / (dfX['HEIGHT3'] ** 2)
  var_dict['BMI'] = {
    'var_type': 'float32',
    'reference_level': '22'
  }

  dfX = dfX.drop(['HEIGHT3', 'WEIGHT2'], axis = 1)
  dfX.loc[dfX['CHILDREN'] > 80, 'CHILDREN'] -= 80

  # alc_per_week = dfX.loc[dfX['ALCDAY4'] < 200, 'ALCDAY4'] - 100
  # alc_per_month = dfX.loc[dfX['ALCDAY4'] >= 200, 'ALCDAY4'] - 200
  # dfX.loc[dfX['ALCDAY4'] < 200, 'ALCDAY4'] = alc_per_week * 4
  # dfX.loc[dfX['ALCDAY4'] >= 200, 'ALCDAY4'] = alc_per_month
  # dfX.loc[dfX['ALCDAY4'] > 30, 'ALCDAY4'] = 30

  #impute missing datas
  for c in dfX.columns:
    if pd.isna(dfX[c]).any():
      var_type = var_dict[c]['var_type']
      reference_level = var_dict[c]['reference_level']
      if var_type == 'float32':
        dfX[c] = dfX[c].fillna(reference_level.astype('float32'))
      elif var_type == 'category':
        dfX[c] = dfX[c].fillna(reference_level)

  #Convert categorical variables
  for c in dfX.columns:
    var_type = var_dict[c]['var_type']
    if var_type == 'category':
      dfX[c] = pd.Categorical(dfX[c], 
                              categories = var_dict[c]['categories'])


  formula = create_formula(dfX.columns, var_dict)

  if response in df.columns:
    var_map = var_dict[response]['var_map']
    response_map = var_dict[response]['response_map']
    dfXy = pd.concat([dfX, df[response]], axis = 1)
    dfXy[response] = dfXy[response].astype(
      'string').replace(var_map).replace(response_map)
    dfXy = dfXy.dropna(axis = 0, subset = [response])
    dfXy = dfXy.drop(dfXy[dfXy[response] == 'NA'].index)
    dfXy[response] = dfXy[response].astype('category')
    formula = response + ' ~ ' + formula

    return dfXy, formula, var_dict
  
  else:
    return dfX, formula, var_dict

def create_formula(columns, var_dict):
  s = ''
  for c in columns:
    column_dict = var_dict[c]
    var_type = column_dict['var_type']
    if var_type == 'category':
      reference_level = column_dict['reference_level']
      s += 'C(' + c + ', Treatment(reference="' + reference_level + '")) + '
    else:
      s += c + ' + '
  return s[:-3]

def load_model(params):
  dfX = pd.DataFrame([np.zeros(len(params))], columns = params.index)
  dfy = pd.DataFrame([0])
  logit_model = sm.GLM(dfy, dfX, family=sm.families.Binomial())
  return logit_model

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

var_dict = json.load(open('var_categories_default.json'))
sections = pd.Series([var_dict[k]['section'] 
                      if 'section' in var_dict[k].keys() else pd.NA 
                      for k in var_dict.keys()])
sections = sections.dropna().unique()
options = []
 
st.header('Smoking Status Prediction Model')

marker_placeholder = st.empty()
marker_placeholder.markdown('Answer the following survey questions to obtain a prediction of the probability that an individual who has smoked 100 or more cigarettes in their lifetime still smokes cigarettes presently.')
placeholders = []
subheaders = []
for section in sections:
  subheader = st.empty()
  subheader.subheader(section, divider = 'rainbow')
  subheaders.append(subheader)
  for c in var_dict.keys():
      if var_dict[c]['var_type'] == 'category' and 'section' in var_dict[c].keys() and var_dict[c]['section'] == section:
        placeholder = st.empty()
        placeholders.append(placeholder)

        responses = list(var_dict[c]['var_map'].values())

        option = placeholder.selectbox(
          var_dict[c]['question'],
          responses,
          index = np.random.randint(0, len(responses)))
        options.append([c, option])
      elif var_dict[c]['var_type'] == 'float32' and 'section' in var_dict[c].keys() and var_dict[c]['section'] == section:
        placeholder = st.empty()
        placeholders.append(placeholder)
        option = placeholder.text_input(var_dict[c]['question'], value = str(var_dict[c]['reference_level']))
        options.append([c, option])

button_placeholder = st.empty()
if button_placeholder.button('Get Prediction'):
  st.session_state['stage'] = 'button_pressed'
  with st.spinner('Wait for it...'):

    input = list(zip(*options))
    input = pd.DataFrame([list(input[1])], columns = list(input[0]), dtype = 'object')
    covariates = [
    '_STATE',
    'GENHLTH','PHYSHLTH','MENTHLTH',
    'PRIMINSR','PERSDOC3','MEDCOST1','CHECKUP1',
    'EXERANY2','SLEPTIM1',
    'LASTDEN4','RMVTETH4',
    'MARITAL','EDUCA','RENTHOM1','VETERAN3','EMPLOY1','CHILDREN',
    'INCOME3','WEIGHT2','HEIGHT3',
    'ALCDAY4',
    'FLUSHOT7','PNEUVAC4','TETANUS1',
    'HIVTST7','HIVRISK5',
    'COVIDPOS'
    ]
    input, formula, var_dict = preprocess(input, var_dict)
    inputX = patsy.dmatrix(formula, input, return_type='dataframe')
    params = pd.read_csv('params_reduced.csv', 
                                  index_col=0).squeeze().rename(None)
    logit_model = load_model(params)
    pred = logit_model.predict(params, inputX)[0]*100

  for placeholder in placeholders:
    placeholder.empty()
  for subheader in subheaders:
    subheader.empty()
  button_placeholder.empty()
  marker_placeholder.empty()
  

  prediction_placeholder = st.empty()
  prediction_placeholder.markdown('The probability that this individual continues to smoke is: ' + str(np.around(pred,2)) + '%. \n\nThe probability for a reference individual is 17.47%. See the plot below for a visualization of the attributes that contribute to this discrepancy. A negative value for an attribute indicates that the response decreases the probability of smoking compared to the reference individual. A positive value indicates that the response increases the probability of smoking compared to the reference individual. Specifically, a value of $x$ indicates an increase an $x$ increase in the log-odds of the individual smoking vs. not smoking, when compared to the reference individual.')

  effects = []
  for c in input.columns:
      reference_level =  var_dict[c]['reference_level']
      var_type =  var_dict[c]['var_type']
      level =  input[c].iloc[0]
      if var_type == 'category':
          if level == reference_level:
              effects.append([c, level, reference_level, 0])
          else:
              param_key = 'C(' + c + ', Treatment(reference="' + reference_level + '"))[T.' + level + ']'
              effects.append([c, level, reference_level, params[param_key]])
      else:
          reference_level = float(reference_level)
          effects.append([c, level, reference_level, params[c]*(level - reference_level)])

  effects = pd.DataFrame(effects, columns = ['Question', 'Response', 'Reference', 'Effect'])

  effects = effects.sort_values(by = 'Effect')
  effects = effects.loc[effects['Effect'] != 0,:]
  effects['Response'] = effects['Question'] + ' = ' + effects['Response'].astype('string')
  fig = px.bar(effects, x = 'Effect', y = 'Response', orientation = 'h', color = 'Response', template='plotly_dark', title='Factors Contributing to Smoking Probability', hover_data = ['Effect', 'Reference'])
  fig.update_layout(showlegend=False)
  event = st.plotly_chart(fig)
  
  button_placeholder = st.empty()
  if button_placeholder.button('Try Again'):
    st.rerun()