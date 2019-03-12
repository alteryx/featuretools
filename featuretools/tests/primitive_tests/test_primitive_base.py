from featuretools.primitives import Max

def test_call():
    primitive = Max()

    #the assert is run twice on purpose
    assert 5 == primitive(range(6))
    assert 5 == primitive(range(6))