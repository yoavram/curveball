import matplotlib.pyplot as plt
import seaborn as sns

def plot_96wells(df, y, x='Time', func=plt.plot, strains=None):
	if strains==None:
		strains=[]
	g = sns.FacetGrid(df, hue='Strain', col='Number', row='Letter', 
                  sharex=True, sharey=True, size=1,
                  aspect=12./8., despine=True,margin_titles=True,
                  hue_order=strains, palette=sns.color_palette("Set1", 4))
	g.map(func, x, y)
	g.fig.set_figwidth(12) 
	g.fig.set_figheight(8)
	plt.locator_params(nbins=4) # 4 ticks is enough
	g.set_axis_labels('','') # remove facets axis labels
	g.fig.text(0.5, 0, x, size='x-large') # xlabel
	g.fig.text(-0.01, 0.5, y, size='x-large', rotation='vertical') # ylabel
	return g
