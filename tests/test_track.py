import numpy as np
import pytest

from racing_gym.track import PRESETS, SPACING, make_track, random_track


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_are_valid_closed_tracks(name):
    t = make_track(name)
    gaps = np.linalg.norm(np.diff(np.vstack([t.centerline, t.centerline[:1]]), axis=0), axis=1)
    assert gaps.max() < SPACING * 1.5, "centreline must be evenly resampled and closed"
    assert t.min_clearance() > t.width, "two parts of the track overlap"
    assert t.mask[int(t.centerline[0, 1]), int(t.centerline[0, 0])], "start must be on the track"
    assert abs(t.length - t.cum_s[-1] - gaps[-1]) < 1e-6


def test_random_tracks_are_deterministic_and_distinct():
    a, b, c = random_track(42), random_track(42), random_track(43)
    np.testing.assert_array_equal(a.centerline, b.centerline)
    assert a.centerline.shape != c.centerline.shape or not np.allclose(a.centerline, c.centerline)


def test_track_spec_errors_are_helpful():
    with pytest.raises(ValueError, match="Presets"):
        make_track("monza")
    with pytest.raises(ValueError, match="random:<int>"):
        make_track("random:abc")


def test_width_override():
    assert make_track("classic", width=120).width == 120
    assert make_track("classic", width=120).mask.sum() > make_track("classic").mask.sum()


def test_nearest_index_local_search_agrees_with_global():
    t = make_track("snake")
    for i in range(0, len(t.centerline), 37):
        x, y = t.centerline[i] + t.normals[i] * 10
        assert t.nearest_index(x, y) == t.nearest_index(x, y, hint=(i + 5) % len(t.centerline))
