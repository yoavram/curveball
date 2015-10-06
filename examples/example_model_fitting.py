import pandas as pd
import curveball
plate = pd.read_csv('plate_templates/G-RG-R.csv')
df = curveball.ioutils.read_tecan_xlsx('data/Tecan_280715.xlsx', label='OD', plate=plate)
models, fig, ax = curveball.models.fit_model(df[df.Strain == 'G'], PLOT=True, PRINT=False)
fig.savefig('docs/_static/example_model_fitting.svg')
