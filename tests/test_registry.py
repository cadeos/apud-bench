from apud_bench import registry


def test_registry_is_consistent():
    assert registry.validate(registry.load()) == []


def test_every_tier_is_populated():
    tiers = {s.tier for s in registry.load()}
    assert tiers == set(registry.TIERS)


def test_validate_rejects_nc_in_core():
    bad = registry.Subset(
        id="x",
        name="x",
        url="https://example.org",
        license="CC-BY-NC-4.0",
        tier="core",
        stratum="medieval",
        script="Latn",
        langs=["lat"],
        centuries="12",
        tasks=["T"],
    )
    errors = registry.validate([bad])
    assert any("not allowed in core" in e for e in errors)
    assert any("outside nc-extension" in e for e in errors)


def test_validate_rejects_duplicates_and_bad_tasks():
    s = registry.load()[0]
    dup = registry.Subset(**{**s.__dict__, "tasks": ["X"]})
    errors = registry.validate([s, dup])
    assert any("duplicate id" in e for e in errors)
    assert any("tasks must be" in e for e in errors)
