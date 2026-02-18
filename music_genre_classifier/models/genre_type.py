from enum import Enum


class GenreType(Enum):
    POP = "pop"
    ROCK = "rock"
    ELETRONICA = "eletronica"
    CLASSICA = "classica"
    FORRO = "forro"

    @staticmethod
    def to_label(genre):
        return list(GenreType).index(genre)

    @staticmethod
    def from_label(label):
        return list(GenreType)[label]

    @staticmethod
    def get_name(label):
        return list(GenreType)[label].value
