import pandas as pd
import pytest

from featuretools import Feature, calculate_feature_matrix
from featuretools.demo import load_mock_customer


def test_datetime_plus_timedelta():
    """Test adding a timedelta to a datetime feature"""
    es = load_mock_customer(return_entityset=True)
    date_of_birth = Feature(es["customers"]["date_of_birth"])
    
    # Add 1 year timedelta
    feature = date_of_birth + pd.Timedelta(days=365)
    
    # Should not raise AssertionError
    df = calculate_feature_matrix([feature], es, instance_ids=[1, 2])
    assert df is not None
    assert len(df) == 2
    assert feature.get_name() in df.columns


def test_datetime_minus_timedelta():
    """Test subtracting a timedelta from a datetime feature"""
    es = load_mock_customer(return_entityset=True)
    date_of_birth = Feature(es["customers"]["date_of_birth"])
    
    # Subtract 1 year timedelta
    feature = date_of_birth - pd.Timedelta(days=365)
    
    # Should not raise AssertionError
    df = calculate_feature_matrix([feature], es, instance_ids=[1, 2])
    assert df is not None
    assert len(df) == 2
    assert feature.get_name() in df.columns


def test_datetime_plus_timedelta_calculation():
    """Test that datetime + timedelta produces correct results"""
    es = load_mock_customer(return_entityset=True)
    date_of_birth = Feature(es["customers"]["date_of_birth"])
    
    # Add 1 day
    feature = date_of_birth + pd.Timedelta(days=1)
    
    df = calculate_feature_matrix([date_of_birth, feature], es, instance_ids=[1])
    
    # The result should be one day after the original date
    original_date = df[date_of_birth.get_name()].iloc[0]
    result_date = df[feature.get_name()].iloc[0]
    
    assert (result_date - original_date) == pd.Timedelta(days=1)


def test_datetime_minus_timedelta_calculation():
    """Test that datetime - timedelta produces correct results"""
    es = load_mock_customer(return_entityset=True)
    date_of_birth = Feature(es["customers"]["date_of_birth"])
    
    # Subtract 1 day
    feature = date_of_birth - pd.Timedelta(days=1)
    
    df = calculate_feature_matrix([date_of_birth, feature], es, instance_ids=[1])
    
    # The result should be one day before the original date
    original_date = df[date_of_birth.get_name()].iloc[0]
    result_date = df[feature.get_name()].iloc[0]
    
    assert (original_date - result_date) == pd.Timedelta(days=1)
