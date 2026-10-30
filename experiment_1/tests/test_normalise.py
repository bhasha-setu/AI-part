from guided_practice.normalise import normalise


def test_lowercase_trim_collapse():
    assert normalise("  My   Name  ") == "my name"


def test_strip_trailing_punctuation():
    assert normalise("I am agree.") == "i am agree"
    assert normalise("Hello!!!") == "hello"


def test_strip_leading_fillers():
    assert normalise("um my name is ravi") == "my name is ravi"
    assert normalise("Uh, I'm from Delhi") == "i am from delhi"


def test_contraction_expansion():
    assert normalise("I'm from Dehradun") == "i am from dehradun"
    assert normalise("I'd like a coffee") == "i would like a coffee"


def test_curly_apostrophe():
    assert normalise("I’m from Delhi") == "i am from delhi"


def test_internal_comma_removed():
    assert normalise("Can I have the menu, please?") == "can i have the menu please"


def test_typo_passes_through_unchanged():
    # Normalisation alone does NOT fix spelling; typo tolerance is separate.
    assert normalise("my nam is ravi") == "my nam is ravi"


def test_empty_and_none():
    assert normalise("") == ""
    assert normalise(None) == ""
    assert normalise("   ") == ""
