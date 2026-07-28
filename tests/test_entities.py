import pytest

from localscrub import SAFE_HARBOR, Entity, EntityType, Stage


def test_safe_harbor_has_all_18_categories():
    assert len(SAFE_HARBOR) == 18
    assert len(set(SAFE_HARBOR)) == 18


def test_safe_harbor_order_matches_regulation():
    # (A) names first, (R) catch-all last
    assert SAFE_HARBOR[0] is EntityType.NAME
    assert SAFE_HARBOR[-1] is EntityType.OTHER_UNIQUE_ID


def test_generic_pii_is_outside_safe_harbor():
    generic = set(EntityType) - set(SAFE_HARBOR)
    assert generic == {
        EntityType.CREDIT_CARD,
        EntityType.PASSPORT,
        EntityType.NATIONAL_ID,
        EntityType.SECRET,
    }


def test_entity_valid():
    e = Entity(
        type=EntityType.SSN,
        start=10,
        end=21,
        text="123-45-6789",
        confidence=0.99,
        stage=Stage.RULES,
    )
    assert e.text == "123-45-6789"


@pytest.mark.parametrize(
    ("start", "end", "text", "confidence"),
    [
        (5, 5, "", 0.5),  # empty span
        (-1, 3, "abcd", 0.5),  # negative start
        (0, 3, "abcd", 0.5),  # text/span length mismatch
        (0, 4, "abcd", 1.5),  # confidence out of range
    ],
)
def test_entity_rejects_invalid(start, end, text, confidence):
    with pytest.raises(ValueError):
        Entity(
            type=EntityType.NAME,
            start=start,
            end=end,
            text=text,
            confidence=confidence,
            stage=Stage.LLM,
        )
