from excel_table_cnn.evaluation.eob import eob, eob_precision_recall, match_detections


def test_eob_exact_match():
    assert eob([0, 0, 3, 3], [0, 0, 3, 3]) == 0


def test_eob_is_max_deviation():
    assert eob([0, 0, 5, 3], [0, 0, 3, 3]) == 2
    assert eob([1, 0, 4, 3], [0, 0, 3, 3]) == 1
    assert eob([2, 5, 9, 20], [3, 5, 11, 21]) == 2


def test_match_exact_detections():
    gts = [[0, 0, 3, 3], [10, 10, 20, 20]]
    preds = [[0, 0, 3, 3], [10, 10, 20, 20]]
    scores = [0.9, 0.8]
    assert match_detections(preds, scores, gts, threshold=0) == (2, 0, 0)


def test_duplicate_detection_counts_as_fp():
    gts = [[0, 0, 3, 3]]
    preds = [[0, 0, 3, 3], [0, 0, 3, 3]]
    scores = [0.9, 0.8]
    assert match_detections(preds, scores, gts, threshold=0) == (1, 1, 0)


def test_missed_gt_counts_as_fn():
    gts = [[0, 0, 3, 3], [10, 10, 20, 20]]
    preds = [[0, 0, 3, 3]]
    scores = [0.9]
    assert match_detections(preds, scores, gts, threshold=0) == (1, 0, 1)


def test_near_miss_passes_only_looser_threshold():
    gts = [[0, 0, 3, 3]]
    preds = [[0, 0, 5, 3]]  # EoB = 2
    scores = [0.9]
    assert match_detections(preds, scores, gts, threshold=0) == (0, 1, 1)
    assert match_detections(preds, scores, gts, threshold=2) == (1, 0, 0)


def test_precision_recall_aggregates_over_sheets():
    per_sheet_preds = [
        ([[0, 0, 3, 3]], [0.9]),          # sheet 1: exact hit
        ([[0, 0, 9, 9], [40, 40, 50, 50]], [0.8, 0.7]),  # sheet 2: 1 hit + 1 FP
        ([], []),                          # sheet 3: missed its table
    ]
    per_sheet_gts = [
        [[0, 0, 3, 3]],
        [[0, 0, 9, 9]],
        [[5, 5, 8, 8]],
    ]
    report = eob_precision_recall(per_sheet_preds, per_sheet_gts, thresholds=(0, 2))
    assert report["eob0"]["tp"] == 2
    assert report["eob0"]["fp"] == 1
    assert report["eob0"]["fn"] == 1
    assert report["eob0"]["precision"] == 2 / 3
    assert report["eob0"]["recall"] == 2 / 3
    assert report["eob2"] == report["eob0"]  # same outcome at looser threshold here


def test_no_predictions_no_gt():
    report = eob_precision_recall([([], [])], [[]], thresholds=(0,))
    assert report["eob0"]["precision"] == 0.0
    assert report["eob0"]["recall"] == 0.0
