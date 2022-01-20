import pickle
import numpy as np
import pandas as pd
import math
model = pickle.load(open('abalome.pkl', 'rb'))
sex_encod = pickle.load(open('sex_lbl.pkl', 'rb'))
def predict(df):
    df = df[['sex', 'length', 'diameter','height', 'whole-weight', 'shucked-weight','viscera-weight','shell-weight']]
    df.sex = sex_encod.transform(df.sex)
    predictions = model.predict(df)
    return list(np.floor(predictions))


