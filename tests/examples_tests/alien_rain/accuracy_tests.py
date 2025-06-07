import numpy as np
import pytest

class TestNPYComparison:
    @staticmethod
    def calculate_mse_npy(file_path1, file_path2):
        try:
            data1 = np.load(file_path1)
            data2 = np.load(file_path2)
        except FileNotFoundError as e:
            pytest.fail(f"Error loading NPY file: {e}")
        except Exception as e:
            pytest.fail(f"An error occurred while loading NPY files: {e}")

        if data1.shape != data2.shape:
            pytest.fail(f"NPY files have different shapes: {data1.shape} vs {data2.shape}")

        mse = np.mean((data1 - data2)**2)
        return mse

    def test_compare_npy_files_mse(self):
        # TODO: Replace with the actual paths to your NPY files
        file1 = "path/to/your/first_file.npy"
        file2 = "path/to/your/second_file.npy"

        tolerance = 1e-4  # Adjust tolerance as needed
        mse = self.calculate_mse_npy(file1, file2)
        assert mse < tolerance, f"MSE {mse} exceeds tolerance {tolerance}"