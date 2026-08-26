from app.evaluation.cost import estimate_cost_usd, lookup_rates


def test_known_model_cost():
    rates = lookup_rates("gpt-4o-mini")
    assert rates is not None
    cost = estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == rates[0] + rates[1]


def test_unknown_model_is_null_not_invented():
    assert lookup_rates("definitely-not-a-real-model-xyz") is None
    assert estimate_cost_usd("definitely-not-a-real-model-xyz", 100, 100) is None


def test_missing_tokens_is_null():
    assert estimate_cost_usd("gpt-4o-mini", None, 10) is None
