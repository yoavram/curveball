import unittest
from curveball import Plate
import numpy as np
import tempfile
import os
import filecmp
import seaborn as sns

fname = 'tests/plate.csv'


class PlateTest(unittest.TestCase):
	def setUp(self):
		self.array = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0]])
		np.savetxt(fname, self.array, fmt='%s', delimiter=', ')


	def tearDown(self):
		os.remove(fname)


	def test_init(self):
		plate = Plate(self.array)
		self.assertTrue((plate._array == self.array).all())


	def test_repr(self):
		plate = Plate(self.array)
		self.assertEquals(str(plate), str(self.array))


	def test_from_csv(self):
		plate  = Plate.from_csv(fname)
		self.assertTrue((plate._array == self.array).all())


	def test_to_csv(self):
		plate = Plate(self.array)
		output = tempfile.NamedTemporaryFile(delete=False)
		plate.to_csv(output.name)
		self.assertTrue(filecmp.cmp(output.name, fname))
		output.close()	
		os.remove(output.name)


	def test_to_array(self):
		plate  = Plate(self.array)
		arr = plate.to_array()
		self.assertTrue((arr == self.array).all())
		self.assertFalse(arr is self.array)


	def test_get_strains(self):
		plate  = Plate(self.array)
		strains = plate.strains
		self.assertEquals(strains,[0,1,2,3,4])


	def test_colormap(self):
		plate  = Plate(self.array)
		colormap = plate.colormap
		self.assertEquals(sorted(colormap.keys()), plate.strains)
		self.assertEquals(colormap[0], Plate.BLANK_COLOR)
		colors = sorted(colormap.values())
		colorset = sorted(set(colors))
		self.assertEquals(colors, colorset)


	def test_strain2wells(self):
		plate  = Plate(self.array)
		self.assertEquals(plate.strain2wells(1), ('A1','A2','B1','B2'))
		self.assertEquals(plate.strain2wells(2), ('C3','C4','D3','D4'))
		self.assertEquals(plate.strain2wells(3), ('E5','E6','F5','F6'))
		self.assertEquals(plate.strain2wells(4), ('G7','G8','H7','H8'))


	def test_strain2color(self):
		palette = sns.color_palette("Set1", 4)
		plate  = Plate(self.array, palette=palette)
		self.assertEquals(plate.strain2color(1), palette[0])
		self.assertEquals(plate.strain2color(2), palette[1])
		self.assertEquals(plate.strain2color(3), palette[2])
		self.assertEquals(plate.strain2color(4), palette[3])


	def test_well2strain(self):
		plate  = Plate(self.array)
		self.assertEquals(plate.well2strain('A1'), 1)
		self.assertEquals(plate.well2strain('C3'), 2)
		self.assertEquals(plate.well2strain('E5'), 3)
		self.assertEquals(plate.well2strain('G7'), 4)
		self.assertEquals(plate.well2strain('G12'), 0)


	def test_well2color(self):
		palette = sns.color_palette("Set1", 4)
		plate  = Plate(self.array, palette=palette)
		self.assertEquals(plate.well2color('A1'), palette[0])
		self.assertEquals(plate.well2color('C3'), palette[1])
		self.assertEquals(plate.well2color('E5'), palette[2])
		self.assertEquals(plate.well2color('G7'), palette[3])
		self.assertEquals(plate.well2color('G12'), Plate.BLANK_COLOR)


if __name__ == '__main__':
    unittest.main()