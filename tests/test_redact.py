from localscrub.entities import Entity, EntityType, Stage
from localscrub.redact import apply_redactions


def entity(entity_type, start, text, confidence=0.9):
    return Entity(
        type=entity_type,
        start=start,
        end=start + len(text),
        text=text,
        confidence=confidence,
        stage=Stage.RULES,
    )


def test_single_replacement():
    text = "mail me@example.com now"
    redacted, placeholders = apply_redactions(text, [entity(EntityType.EMAIL, 5, "me@example.com")])
    assert redacted == "mail [EMAIL_1] now"
    assert placeholders == {5: "[EMAIL_1]"}


def test_multiple_entities_offsets_stay_correct():
    text = "a@example.com and b@example.org"
    entities = [
        entity(EntityType.EMAIL, 0, "a@example.com"),
        entity(EntityType.EMAIL, 18, "b@example.org"),
    ]
    redacted, _ = apply_redactions(text, entities)
    assert redacted == "[EMAIL_1] and [EMAIL_2]"


def test_repeated_text_gets_same_placeholder():
    text = "a@example.com then a@example.com"
    entities = [
        entity(EntityType.EMAIL, 0, "a@example.com"),
        entity(EntityType.EMAIL, 19, "a@example.com"),
    ]
    redacted, _ = apply_redactions(text, entities)
    assert redacted == "[EMAIL_1] then [EMAIL_1]"


def test_counters_are_per_type():
    text = "a@example.com 192.0.2.7"
    entities = [
        entity(EntityType.EMAIL, 0, "a@example.com"),
        entity(EntityType.IP_ADDRESS, 14, "192.0.2.7"),
    ]
    redacted, _ = apply_redactions(text, entities)
    assert redacted == "[EMAIL_1] [IP_ADDRESS_1]"


def test_adjacent_entities():
    text = "ab"
    entities = [entity(EntityType.NAME, 0, "a"), entity(EntityType.NAME, 1, "b")]
    redacted, _ = apply_redactions(text, entities)
    assert redacted == "[NAME_1][NAME_2]"


def test_unnumbered_mode():
    text = "a@example.com and b@example.org"
    entities = [
        entity(EntityType.EMAIL, 0, "a@example.com"),
        entity(EntityType.EMAIL, 18, "b@example.org"),
    ]
    redacted, placeholders = apply_redactions(text, entities, numbered=False)
    assert redacted == "[EMAIL] and [EMAIL]"
    assert set(placeholders.values()) == {"[EMAIL]"}


def test_input_order_does_not_matter():
    text = "a@example.com and b@example.org"
    entities = [
        entity(EntityType.EMAIL, 18, "b@example.org"),
        entity(EntityType.EMAIL, 0, "a@example.com"),
    ]
    redacted, _ = apply_redactions(text, entities)
    assert redacted == "[EMAIL_1] and [EMAIL_2]"
