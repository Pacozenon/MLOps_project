#!/usr/bin/env python

"""Tests for `mlops_project` package."""

# Change current directory to /Users/francisco.torres/Documents/GitHub/MLOps_project/refactor/mlops_project/tests

import os

import pandas as pd
import pytest

from mlops_project.preprocess.preprocess_data import Change_TransactionType


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_change_attribute():
    """
    Test the `transform` method of the type attribute.

    This test checks if the transformer correctly translate from categorical to numerical
    """
    # Sample DataFrame with missing values
    data = {
        "type": [
            "CASH_OUT",
            "PAYMENT",
            "CASH_OUT",
            "PAYMENT",
            "TRANSFER",
            "DEBIT",
            "TRANSFER",
        ]
    }
    df = pd.DataFrame(data)

    # Instantiate the custom transformer with specified variables
    changetype = Change_TransactionType()

    # Transform the DataFrame using the custom transformer
    df_transformed = changetype.transform(df)

    # Check if the transformed DataFrame has the expected additional columns
    expected_columns = [1, 2, 1, 2, 4, 5, 4]
    assert all(
        col in df_transformed.columns for col in expected_columns
    ), f"The transformed DataFrame should have the following additional columns: {expected_columns}"


def does_csv_file_exist(file_path):
    """
    Check if a CSV file exists at the specified path.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def test_csv_file_existence():
    """
    Test case to check if the CSV file exists.
    """
    # Provide the path to your CSV file that needs to be tested
    # refactored folder
    REFACTORED_DIRECTORY = "/Users/francisco.torres/Documents/GitHub/MLOps_project/Refactor/mlops_project/mlops_project"

    os.chdir(REFACTORED_DIRECTORY)
    csv_file_path = "./data/retrieved_data.csv"

    DATASETS_DIR = "./data/"

    # Call the function to check if the CSV file exists
    file_exists = does_csv_file_exist(csv_file_path)
    # Use Pytest's assert statement to check if the file exists
    assert file_exists == True, f"The CSV file at '{csv_file_path}' does not exist."


if __name__ == "__main__":
    # Run the test function using Pytest
    pytest.main([__file__])
