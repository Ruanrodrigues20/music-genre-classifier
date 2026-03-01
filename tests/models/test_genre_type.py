import pytest
from music_genre_classifier.models.genre_type import GenreType

def test_enum_values():
    assert GenreType.POP.value == 0
    assert GenreType.ROCK.value == 1
    assert GenreType.ELETRONICA.value == 2
    assert GenreType.CLASSICA.value == 3
    assert GenreType.FORRO.value == 4


def test_to_label():
    assert GenreType.to_label(GenreType.POP) == 0
    assert GenreType.to_label(GenreType.ROCK) == 1


def test_from_label():
    assert GenreType.from_label(0) == GenreType.POP
    assert GenreType.from_label(3) == GenreType.CLASSICA


def test_get_name():
    assert GenreType.get_name(0) == "pop"
    assert GenreType.get_name(1) == "rock"
    assert GenreType.get_name(4) == "forro"


def test_from_label_invalid():
    with pytest.raises(ValueError):
        GenreType.from_label(99)