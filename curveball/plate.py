"""
Matplotlib microplate visualization
----------------------
A simple microplate implementation in matplotlib.
Based on https://jakevdp.github.io/blog/2012/12/06/minesweeper-in-matplotlib/

Author: Yoav Ram <yoavram@gmail.com>, Mar. 2015
License: BSD
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

class Plate(object):
	edge_color = '#888888'    
	all_colors = ('none', 'blue', 'green', 'red', 'darkblue', 'darkred', 'darkgreen', 'black', 'black')

	@classmethod
	def ninety_six_well(cls, nstrains=2):
		return cls(12, 8, nstrains)

	@classmethod
	def from_csv(cls, filename):
		strains = np.loadtxt(filename, dtype=int, delimiter=', ')
		strains = np.rot90(strains, 3)
		nstrains = len(np.unique(strains))
		if not 0 in strains:
			nstrains += 1            
		plate = cls(strains.shape[0], strains.shape[1], nstrains)
		plate.strains = strains
		for i in range(plate.width):
			for j in range(plate.height):
				col = strains[i,j]
				if col > 0:
					col = plate.colors[col]                
					plate.squares[i,j].set_facecolor(col)   
		return plate
	

	def to_csv(self, fname):        
		return np.savetxt(fname, np.rot90(self.strains), fmt='%d', delimiter=', ')


	def __repr__(self):
		return str(np.rot90(self.strains))
  

	def __init__(self, width, height, nstrains):
		self.width, self.height, self.nstrains = width, height, nstrains+1     

		self.colors = self.all_colors[:self.nstrains]
	
		# Create the figure and axes
		self.fig = plt.figure(figsize=((width + 2) / 3., (height + 2) / 3.))
		self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9),
									aspect='equal', frameon=False,
									xlim=(-0.05, width + 0.05),
									ylim=(-0.05, height + 0.05))
		for axis in (self.ax.xaxis, self.ax.yaxis):
			axis.set_major_formatter(plt.NullFormatter())
			axis.set_major_locator(plt.NullLocator())

		# Create the grid of squares
		self.squares = np.array([[RegularPolygon((i + 0.5, j + 0.5),
												 numVertices=4,
												 radius=0.5 * np.sqrt(2),
												 orientation=np.pi / 4,
												 ec=self.edge_color,
												 fc=self.colors[0])
								  for j in range(height)]
								 for i in range(width)])
		[self.ax.add_patch(sq) for sq in self.squares.flat]  
		self.strains = np.zeros((width, height), dtype=int)
		# Create event hook for mouse clicks
		self.fig.canvas.mpl_connect('button_press_event', self._button_press)
				
	
	def _click_square(self, i, j):        
		col = self.strains[i,j]
		col = (col + 1) % self.nstrains
		self.strains[i,j] = col
		col = self.colors[col]                
		self.squares[i,j].set_facecolor(col)


	def _button_press(self, event):
		if (event.xdata is None) or (event.ydata is None):
			return
		i, j = map(int, (event.xdata, event.ydata))
		if (i < 0 or j < 0 or i >= self.width or j >= self.height):
			return       
		self._click_square(i, j)
		self.fig.canvas.draw()

if __name__ == '__main__':
	plate = Plate.ninety_six_well(3)
	plt.show()
	fname = raw_input("What should I call it?\n")
	plate.to_csv(fname)