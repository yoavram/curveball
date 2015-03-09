import unittest
from curveball import Plate
import numpy as np


fname = 'tests/strains.csv'
tmpname = 'tests/tmp.csv'


class PlateTest(unittest.TestCase):
	def setUp(self):
		self.strains = np.array([[0, 0, 0, 0, 0, 0, 1, 1],
								[0, 0, 0, 0, 0, 0, 1, 1],
								[0, 0, 0, 0, 2, 2, 0, 0],
								[0, 0, 0, 0, 2, 2, 0, 0],
								[0, 0, 3, 3, 0, 0, 0, 0],
								[0, 0, 3, 3, 0, 0, 0, 0],
								[4, 4, 0, 0, 0, 0, 0, 0],
								[4, 4, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0, 0, 0]])


	def test_from_csv(self):
		plate  = Plate.from_csv(fname)
		self.assertTrue((plate.strains == self.strains).all())


	def test_to_csv(self):
		plate = Plate(self.strains.shape[1], self.strains.shape[0], 5)
		plate.strains = self.strains.copy()
		plate.to_csv(tmpname)
		import filecmp
		self.assertTrue(filecmp.cmp(tmpname, fname))

	def test_to_array(self):
		plate  = Plate.ninety_six_wells(2)
		arr = plate.to_array()
		self.assertTrue((arr == 0).all())
		self.assertEquals(arr.shape, (8,12))

	def test_well2strain(self):
		plate  = Plate.ninety_six_wells(2)
		plate.strains = self.strains
		well2strain = plate.well2strain
		self.assertEquals(well2strain('A1'), 1)
		self.assertEquals(well2strain('A3'), 0)
		self.assertEquals(well2strain('C2'), 0)
		self.assertEquals(well2strain('C3'), 2)
		self.assertEquals(well2strain('G12'), 0)
		self.assertEquals(well2strain('G8'), 4)


if __name__ == '__main__':
    unittest.main()