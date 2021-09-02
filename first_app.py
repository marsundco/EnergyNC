from io import StringIO
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time
import pickle
from sklearn.metrics import mean_absolute_error
from math import sqrt, pow
import matplotlib.pyplot as plt


@st.cache(suppress_st_warning=True)
def progress_comp():
    bar = left_col.progress(0)
    for i in range(100):
        #Update the progress bar with each iteration.
        bar.progress(i+1)
        time.sleep(0.02)
    return None
    
header_pic = Image.open('image001.jpg')
st.image(header_pic, 
    use_column_width='always'
)

"""
# Energy predicion for milling processes based on  NC instructions
Load your gcode-file and get a accurate prediction for the energy consumed in production.
"""

left_col, right_col = st.beta_columns(2)

left_col.header('File upload')
right_col.header('Part specifications')

uploaded_file = left_col.file_uploader(
    'Please upload your gcode-file',
    type = 'csv'
)

if uploaded_file is not None:
     #Add a placeholder
    bar = progress_comp()
    left_col.success('Upload succesful!')
    image = Image.open('bsp2.png')
    left_col.image(image, caption='Geladene Step Datei', use_column_width='always')
    #left_col.write('Calculated!')

form = right_col.form(key = 'my_form')
form.radio(
    'Please select the material the part has to be manufactured of',
    ('Steel', 'Aluminium', 'Plastic')
)

form.selectbox(
    'Please select the machine, on which the part will be manufactured',
    ('Spinner U5630', 'Generic')
)

form.slider(
    'How many parts will be produced in one machining operation',
    min_value=1, 
    max_value=100,
    value=10
)

tolerance = form.select_slider(
    'What are the tolerance requirements of your part?',
    options=['low', 'medium', 'high']
    )

number = form.number_input(
    'Raw Material Dimention X-Direction',
    value=60)

number = form.number_input(
    'Raw Material Dimention Y-Direction',
    value=20)

number = form.number_input(
    'Raw Material Dimention Z-Direction',
    format= '%i',
    value=10)

submit_button = form.form_submit_button(label='Predict Energy time series')

pkl_filename = 'MLmodel.pkl'
with open(pkl_filename, 'rb') as file:
    test_model = pickle.load(file)

if uploaded_file is not None and submit_button:
    df2 = pd.read_csv(uploaded_file, index_col=['Unnamed: 0'])
    for i in range(len(df2)) : 
        if str(df2.loc[i, "Commands"]).strip() == 'G0 G90':
         df2.at[i,'Commands'] = 'G0'
        elif str(df2.loc[i, "Commands"]).strip() == 'G41 G94 G1 G90':
            df2.at[i,'Commands'] = 'G1'
        elif str(df2.loc[i, "Commands"]).strip() == 'M58;':
            df2.at[i,'Commands'] = 'M58,'
        
        elif str(df2.loc[i, "Commands"]).strip() == 'G41':
           df2.at[i,'Commands'] = 'MSG'
        elif str(df2.loc[i, "Commands"]).strip() == 'G94 G3 G90':
            df2.at[i,'Commands'] = 'G2'
    df2 = pd.get_dummies(df2, columns=['Commands', 'D'])
    toPredict = ['ENERGY|x', 'ENERGY|y', 'ENERGY|z', 'ENERGY|S', 'ENERGY|T']
    features = ['delta_X', 'delta_Y', 'delta_Z', 'delta_S', 'F_val', 'S', 'D_W', 'Toolchange', 'TurnOp',
    'Commands_G0',
     'Commands_G0 G40 G60',
    'Commands_G0 M106',
    'Commands_G0 M3',
    'Commands_G09',
    'Commands_G1',
    'Commands_G1 G60',
    'Commands_G2',
    'Commands_G4',
    'Commands_G40',
    'Commands_G41 G1',
    'Commands_G4F1',
    'Commands_G54 G0',
    'Commands_G90',
    'Commands_G91',
    'Commands_G94',
    'Commands_G94 G1 G90',
    'Commands_M168',
    'Commands_M169 M167',
    'Commands_M17',
    'Commands_M27 M28',
    'Commands_M5',
    'Commands_M58,',
    'Commands_M59',
    'Commands_MSG',
    'D_D0',
    'D_D1',
    'D_D=$P_TOOL']

    y2 = df2[toPredict]
    X2 = df2[features]

    predictions = test_model.predict(X2)
    df_pred = pd.DataFrame(data=predictions)

    # X Achse 
    st.header('Vorhersage der Energieverbräuche')
    st.subheader('X Achse')
    chart_x = pd.DataFrame(
        {
            'Messung': y2['ENERGY|x'],
            'Vorhersage': df_pred[0]
        }
    )
    st.line_chart(chart_x)

    # Y Achse

    st.subheader('Y Achse')
    chart_y = pd.DataFrame(
        {
            'Messung': y2['ENERGY|y'],
            'Vorhersage': df_pred[1]
        }
    )
    st.line_chart(chart_y)

    # Z Achse
    st.subheader('Z Achse')
    chart_z = pd.DataFrame(
        {
            'Messung': y2['ENERGY|z'],
            'Vorhersage': df_pred[2]
        }
    )
    st.line_chart(chart_z)

    # Spindle
    st.subheader('Spindel Aggregat')
    chart_S = pd.DataFrame(
        {
            'Messung': y2['ENERGY|S'],
            'Vorhersage': df_pred[3]
        }
    )
    st.line_chart(chart_S)

    # Werkzeug
    st.subheader('Werkzeugwechsel')
    chart_T = pd.DataFrame(
        {
            'Messung': y2['ENERGY|T'],
            'Vorhersage': df_pred[4]
        }
    )
    st.line_chart(chart_T)