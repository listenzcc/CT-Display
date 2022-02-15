# %%
import numpy as np
import pandas as pd
import plotly.express as px

from pathlib import Path

# %%
patient_data = pd.read_csv(Path.cwd().joinpath('../CT-data/patientsData.csv'))
patient_data

# %%
weights_data = pd.read_excel(Path.cwd().joinpath('../CT-data/features.xlsx'))
weights_data

# %%
new_columns = ['{}_{}_{}'.format(a, b, c.split('.')[0]) for a, b, c in zip(
    patient_data.iloc[1], patient_data.iloc[0], patient_data.columns)]

for j, e in enumerate(new_columns):
    if e == 'original_shape_LeastAxis':
        new_columns[j] = 'original_shape_LeastAxisLength'

    if e == 'original_shape_MinorAxis':
        new_columns[j] = 'original_shape_MinorAxisLength'

patient_data_1 = patient_data.copy().iloc[2:]

patient_data_1.columns = new_columns
patient_data_1.index = range(len(patient_data_1))
patient_data_1

# %%

data = []

for j in weights_data.index:
    n = weights_data.loc[j, 'combine']
    d = np.array(patient_data_1[n].map(float))
    w = weights_data.loc[j, 'weight']

    data.append((n, d, w))

data

# %%
y = data[0][1] * 0
for n, d, w in data:
    y += d * w
y

# %%


def _label(e):
    if e == '好':
        return 1
    return 0


label = patient_data_1['custom_custom_预后'].map(_label)
label

# %%
table = pd.DataFrame()
table['label'] = patient_data_1['custom_custom_预后']
table['y'] = y

fig = px.box(table, color='label', y='y')
fig.show()

fig = px.histogram(table, color='label', x='y', barmode='overlay', opacity=0.5)
fig.show()

# %%
