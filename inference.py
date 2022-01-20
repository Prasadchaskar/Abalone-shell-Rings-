import pickle
import numpy as np
import pandas as pd
import math
model = pickle.load(open('Abalone/abalome.pkl', 'rb'))
sex_encod = pickle.load(open('Abalone/sex_lbl.pkl', 'rb'))
def predict(df):
    df = df[['sex', 'length', 'diameter','height', 'whole-weight', 'shucked-weight','viscera-weight','shell-weight','rings']]
    df.sex = sex_encod.transform(df.sex)
    predictions = model.predict(df)
    return list(np.floor(predictions))

sex = 'M'	
length = 91	
diameter = 73	
height = 19	
wholeweight = 102.8	
shuckedweight = 44.9	
visceraweight = 20.2	
shellweight = 30.0
rings = 15	
df = pd.DataFrame({ 
    'sex':[sex],
    'length':[length], 
    'diameter':[diameter], 
    'height':[height],
    'whole-weight':[wholeweight], 
    'shucked-weight':[shuckedweight],
    'viscera-weight':[visceraweight],
    'shell-weight':[shellweight],
    'rings' :[rings]

})
print(predict(df))
