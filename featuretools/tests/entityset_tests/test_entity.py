import pytest
from featuretools import variable_types
from ..testing_utils import make_ecommerce_entityset


@pytest.fixture
def es():
    return make_ecommerce_entityset()


def test_enforces_variable_id_is_str(es):
    assert variable_types.Categorical("1", es["customers"])
    with pytest.raises(AssertionError):
        variable_types.Categorical(1, es["customers"])
