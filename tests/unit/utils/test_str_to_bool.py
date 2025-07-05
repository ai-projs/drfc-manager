import pytest
from drfc_manager.utils.str_to_bool import str2bool


@pytest.mark.parametrize(
    "value,expected",
    [
        ("yes", True),
        ("true", True),
        ("t", True),
        ("1", True),
        ("no", False),
        ("false", False),
        ("0", False),
        (None, False),
        (True, True),
        (False, False),
    ],
)
def test_str2bool(value, expected):
    assert str2bool(value) == expected
