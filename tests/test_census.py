import pandas as pd

from excel_table_cnn.data.census import box_census, collect_box_dims, lattice_coverage
from excel_table_cnn.model.rcnn import DEFAULT_ANCHOR_SIZES, DEFAULT_ASPECT_RATIOS


def toy_markup():
    return pd.DataFrame(
        {"table_range": [["A1:C3", "B2:B21"], ["A1:J2", "bogus", ""]]}
    )


def test_collect_box_dims_parses_and_skips_garbage():
    dims = collect_box_dims(toy_markup())
    assert dims == [(3.0, 3.0), (1.0, 20.0), (10.0, 2.0)]


def test_box_census_percentiles():
    stats = box_census(toy_markup())
    assert stats["width"]["p50"] == 3.0
    assert stats["hw_ratio"]["p99"] == 20.0  # the 1x20 column table


def test_perfect_lattice_covers_exactly():
    dims = [(4.0, 4.0)]
    coverage = lattice_coverage(dims, sizes=[4], ratios=[1.0])
    assert coverage[0.5] == 1.0 and coverage[0.7] == 1.0


def test_default_lattice_covers_the_toy_shapes():
    dims = collect_box_dims(toy_markup())
    coverage = lattice_coverage(
        dims, sizes=DEFAULT_ANCHOR_SIZES[0], ratios=DEFAULT_ASPECT_RATIOS[0]
    )
    assert coverage[0.5] == 1.0  # tall 1x20 boxes must be covered (ratio 22/45)
