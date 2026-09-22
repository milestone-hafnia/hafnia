# COCO 2017 as a public dataset — notes

Scratch notes for the follow-up work. **Delete this file before opening a PR.**

Branch: `worktree-coco-2017-public-dataset`
Worktree: `.claude/worktrees/coco-2017-public-dataset`

## What is done

`HafniaDataset.from_name_public_dataset("coco-2017")` downloads, extracts and converts the public
COCO 2017 dataset — including the human pose keypoints. Only the **validation** split is converted,
as agreed for the first iteration.

- `src/hafnia/dataset/format_conversions/format_coco_2017.py` (new)
  - Download + extraction of the COCO archives with skip-if-already-extracted logic.
  - `COCO_2017_SPLITS` defines **all three splits** (train/validation/test) with their archives and
    label files. `SUPPORTED_SPLITS = [SplitName.VAL]` is the only gate — see "Next steps".
  - Archives are cached in `~/hafnia/public_dataset_downloads/coco-2017/.archives/` and deleted after
    a successful extraction (`remove_archive_after_extraction=True`), since `train2017.zip` is 19 GB.
  - An extraction is only considered complete when a marker file
    (`.archives/<archive>.extracted`) **and** the extracted files exist. This detects an interrupted
    extraction, which would otherwise leave e.g. a half-filled `val2017/` folder that looks complete.
- `src/hafnia/dataset/format_conversions/format_coco.py` (extended)
  - Reads a COCO keypoint label file (`person_keypoints_*.json`) into `Skeleton` primitives:
    `read_coco_keypoint_label_file`, `CocoKeypointLabels`, `skeleton_task_from_coco_keypoint_categories`.
  - `CocoSplitPaths.path_keypoints_json` is the new optional input. The Roboflow layout is unchanged.
- `src/hafnia/dataset/format_conversions/public_datasets.py` (new)
  - `public_dataset_to_hafnia_converters()` = torchvision converters + `coco-2017`.
    `HafniaDataset.from_name_public_dataset` now uses this registry.
- `src/hafnia/utils.py`: `get_path_public_dataset_downloads()` → `~/hafnia/public_dataset_downloads`.
- `KeyPoint.labeled` (new field, default `True`) — see "Keypoint visibility" below.
  Dataset format version bumped `0.3.2` → `0.3.3`.
- Tests: `tests/unit/dataset/format_conversions/test_format_coco_2017.py` (9 offline tests, downloads
  covered with `file://` archives), the `tests/data/dataset_formats/format_coco_2017/` fixture (2 real
  val2017 images; one person has 15 labeled keypoints, one has 0), two drawing tests in
  `tests/unit/dataset/test_shape_primitives.py`, and
  `tests/integration/test_coco_2017_public_dataset.py` (`@pytest.mark.slow`, skipped in CI).

## Verified

- Real download + extraction + conversion of the **full validation split**:
  5000 samples, 36781 `Bbox` and 36781 `Bitmask` annotations over 80 classes, 6352 `Skeleton`
  annotations, 107984 keypoints (68215 labeled / 39769 unlabeled). Those counts match the official
  COCO val2017 statistics. `check_dataset(check_splits=False)` passes.
- Re-running skips both downloads and extractions.
- `ruff check`/`format`, `mypy src/ tests/`, `pytest -m "not slow"` → 447 passed, 8 skipped.
  `tests/integration/test_bring_your_own_data.py::test_remove_leftover_integration_test_datasets`
  fails, but it is unrelated: it scans the *platform* for stale `integration-test-dataset-*` datasets
  and asks for manual cleanup.
- Download speed was ~10 MB/s (815 MB val images in ~85 s), not the ~300 KB/s seen in one early
  attempt.

## Heads-up for review

- **The frontend metadata schema changed.** `tests/data/dataset_image_metadata_schema.yaml` gained the
  `labeled` property of `KeyPoint`. The test guarding that file says to notify the front-end team.
  The change is additive with a default, so it is backwards compatible.
- `ARCHIVE_IMAGES_VAL.md5` is `442b8da7639aecaf257c1dceb8ba8c80`, confirmed by downloading the archive
  and by `unzip -t` (no CRC errors). The md5 of `annotations_trainval2017.zip` and
  `image_info_test2017.zip` were confirmed from their S3 ETags. `train2017.zip` and `test2017.zip` are
  multipart uploads with no published checksum, so their `md5` is `None` and their content is only
  verified by the CRC checks during extraction.

## Keypoint visibility

COCO labels a varying number of the 17 body keypoints per person (37% of the keypoints in val2017 are
unlabeled), but hafnia requires a `Skeleton` to carry every keypoint of its class template
(`check_dataset_skeletons`). All 17 keypoints are therefore kept, and:

- `KeyPoint.labeled` is `False` for keypoints that COCO has not labeled. `KeyPoint.draw` skips them
  and `Skeleton.draw` skips the edges towards them, so a pose is no longer drawn with edges running
  to the top-left corner of the image.
- The raw COCO visibility (`0`=not labeled, `1`=labeled but not visible, `2`=visible) is kept in
  `KeyPoint.meta["visibility"]`, as `labeled` does not distinguish occluded from visible keypoints.

## Next steps

- Enable the remaining splits by extending `SUPPORTED_SPLITS` in `format_coco_2017.py`. Things to
  settle when doing that:
  - **The test split has no public annotations.** `image_info_test2017.json` contains `images` and
    `categories` but no `annotations`, so the existing importer produces samples without primitives.
    `check_dataset_tasks` requires at least one annotation per task, so a train+val+test dataset will
    need either a special case for the test split or a documented "no annotations" behaviour.
  - `train2017.zip` is 19 GB (~118k images). The conversion loads `instances_train2017.json`
    (~450 MB) and `person_keypoints_train2017.json` fully into memory.
  - `n_samples` is split evenly across splits by `from_coco_dataset_by_split_definitions` and takes
    the **first** N images of each split, not a random sample.
- A leftover partial download from an early checksum experiment may still be in
  `~/hafnia/coco_downloads_check/` (~255 MB, incomplete). Safe to delete — I was denied permission.
