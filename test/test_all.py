import unittest
import numpy as np
from src.chamfer import compute_symmetric_chamfer_distance
from src.emd import compute_emd


class Tests(unittest.TestCase):
    @staticmethod
    def prepare_test_data(level=1):
        if level == 1:
            gt_points = np.array([[1., 1.], [2., 2.], [3., 3.]])
            predicted_points = gt_points + 1.0
        elif level == 2:
            gt_points = np.array([[1., 1., 1.], [2., 2., 2.]])
            predicted_points = gt_points + 1.0
        elif level == 3:
            gt_points = np.array([[0.5, 3], [1, 4], [2, 4], [3, 5], [5, 5], [5, 2]])
            predicted_points = np.array([[2, 3], [3, 3], [5, 4], [4, 2]])
        else:
            raise ValueError

        return gt_points, predicted_points

    def test_chamfer_1(self):
        gt_points, predicted_points = self.prepare_test_data(level=1)
        true_answer = 4 / 3
        self.assertAlmostEqual(compute_symmetric_chamfer_distance(gt_points, predicted_points), true_answer)

    def test_chamfer_2(self):
        gt_points, predicted_points = self.prepare_test_data(level=2)
        true_answer = 3.0
        self.assertAlmostEqual(compute_symmetric_chamfer_distance(gt_points, predicted_points), true_answer)

    def test_chamfer_3(self):
        gt_points, predicted_points = self.prepare_test_data(level=3)
        true_answer = 3.125
        self.assertAlmostEqual(compute_symmetric_chamfer_distance(gt_points, predicted_points), true_answer)

    def test_emd_1(self):
        gt_points, predicted_points = self.prepare_test_data(level=1)
        true_answer = 2.0
        self.assertAlmostEqual(compute_emd(gt_points, predicted_points), true_answer)

    def test_emd_2(self):
        gt_points, predicted_points = self.prepare_test_data(level=2)
        true_answer = 3.0
        self.assertAlmostEqual(compute_emd(gt_points, predicted_points), true_answer)


if __name__ == '__main__':
    unittest.main()
