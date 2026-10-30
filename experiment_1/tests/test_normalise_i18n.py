"""Script-aware normalisation: Latin and Devanagari through the same code path."""

import unicodedata

from guided_practice.config import load_language
from guided_practice.normalise import normalise

EN = load_language("en")
HI = load_language("hi")


def test_devanagari_danda_is_stripped():
    assert normalise("मेरा नाम राम है।", HI) == "मेरा नाम राम है"


def test_devanagari_matras_and_virama_survive():
    # The old Latin-only regex deleted every one of these.
    assert normalise("नमस्ते", HI) == "नमस्ते"
    assert normalise("मैं दिल्ली से हूँ", HI) == "मैं दिल्ली से हूँ"


def test_nfc_and_nfd_inputs_normalise_to_the_same_string():
    # Latin: e + combining acute (NFD) vs precomposed é (NFC).
    nfd = unicodedata.normalize("NFD", "café")
    nfc = unicodedata.normalize("NFC", "café")
    assert nfd != nfc
    assert normalise(nfd, EN) == normalise(nfc, EN) == "café"


def test_nfc_and_nfd_agree_for_devanagari_nukta():
    # U+0958 (क़) is composition-excluded, so NFC and NFD forms still differ in code
    # points. Normalisation must give one answer for both.
    s = "क़मर"
    nfd = unicodedata.normalize("NFD", s)
    nfc = unicodedata.normalize("NFC", s)
    assert normalise(nfd, HI) == normalise(nfc, HI)


def test_contractions_are_language_specific():
    assert normalise("I'm from Delhi", EN) == "i am from delhi"
    # The Hindi config has no English contraction table, so "i'm" is left alone.
    assert normalise("I'm from Delhi", HI) == "i'm from delhi"


def test_english_fillers_do_not_apply_to_hindi():
    assert normalise("um my name", EN) == "my name"
    assert normalise("um my name", HI) == "um my name"


def test_hindi_placeholder_filler_is_dropped():
    assert normalise("अं मेरा नाम राम है", HI) == "मेरा नाम राम है"


def test_curly_apostrophe_maps_to_straight_only_where_configured():
    assert normalise("I’m fine", EN) == "i am fine"
    # Hindi configures no apostrophe variants, so the curly apostrophe is ordinary
    # punctuation and is stripped. That is intended.
    assert normalise("I\u2019m fine", HI) == "i m fine"
