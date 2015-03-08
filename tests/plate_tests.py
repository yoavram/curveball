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


if __name__ == '__main__':
    unittest.main()