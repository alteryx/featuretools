from featuretools.utils.description_utils import convert_to_nth


def test_first():
    assert convert_to_nth(1) == "1st"
    assert convert_to_nth(21) == "21st"
    assert convert_to_nth(131) == "131st"


def test_second():
    assert convert_to_nth(2) == "2nd"
    assert convert_to_nth(22) == "22nd"
    assert convert_to_nth(232) == "232nd"


def test_third():
    assert convert_to_nth(3) == "3rd"
    assert convert_to_nth(23) == "23rd"
    assert convert_to_nth(133) == "133rd"


def test_nth():
    assert convert_to_nth(4) == "4th"
    assert convert_to_nth(11) == "11th"
    assert convert_to_nth(12) == "12th"
    assert convert_to_nth(13) == "13th"
    assert convert_to_nth(111) == "111th"
    assert convert_to_nth(112) == "112th"
    assert convert_to_nth(113) == "113th"
