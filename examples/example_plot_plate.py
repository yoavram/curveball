import pandas as pd
import curveball

if __name__ == '__main__':
	plate = pd.read_csv('plate_templates/G-RG-R.csv')
	fig, ax = curveball.plots.plot_plate(plate)	
	fig.savefig('docs/_static/example_plot_plate.svg')