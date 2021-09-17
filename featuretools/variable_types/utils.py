from warnings import warn

from woodwork import list_logical_types


def list_variable_types():
    """
    Retrieves all logical types as a dataframe

    Args:
        None

    Returns:
        logical_types (pd.DataFrame): a DataFrame with all logical types
    """
    message = 'list_variable_types has been deprecated. Please use featuretools.list_logical_types instead.'
    warn(message=message, category=FutureWarning)
    return list_logical_types()
