from enum import Enum


class GenreType(Enum):
    POP = 0
    ROCK = 1
    ELETRONICA = 2
    CLASSICA = 3
    FORRO = 4

    @staticmethod
    def to_label(genre):
        return genre.value

    @staticmethod
    def from_label(label):
        return GenreType(label)

    @staticmethod
    def get_name(label):
        return GenreType(label).name.lower()
