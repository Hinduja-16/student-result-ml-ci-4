import os
import unittest

class TestMLPipeline(unittest.TestCase):
    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))
        self.assertTrue(os.path.exists("metrics.json"))
        self.assertTrue(os.path.exists("student_results.csv"))

if __name__ == "__main__":
    unittest.main()
