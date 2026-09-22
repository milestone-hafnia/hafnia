# COCO 2017 as a public dataset — handover notes

Scratch notes for continuing the work. **Delete this file before opening a PR.**

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
- Tests
  - `tests/unit/dataset/format_conversions/test_format_coco_2017.py` — 9 offline tests. Downloads are
    tested with `file://` archives, so no network is needed.
  - `tests/data/dataset_formats/format_coco_2017/` — tiny fixture in the original COCO folder layout
    (2 real val2017 images + annotations cut down to those images). One person annotation has 15
    labeled keypoints, the other has 0, which covers both conversion paths.
  - `tests/integration/test_coco_2017_public_dataset.py` — `@pytest.mark.slow`, skipped in GitHub
    Actions. Downloads ~1 GB on the first run.

Verified: `ruff check`/`format`, `mypy src/ tests/` (via `uvx mypy==1.16.0`), and
`uv run pytest tests -m "not slow"` → 445 passed, 8 skipped. The single failure,
`tests/integration/test_bring_your_own_data.py::test_remove_leftover_integration_test_datasets`, is a
pre-existing leftover-dataset-on-platform failure unrelated to this branch.

## Not verified yet — do this first

1. **Run the real conversion end-to-end.** It has never been run against the real download:

   ```bash
   uv run pytest tests/integration/test_coco_2017_public_dataset.py -q -m slow
   # or
   uv run python -c "from hafnia.dataset.hafnia_dataset import HafniaDataset as D; print(D.from_name_public_dataset('coco-2017', n_samples=20))"
   ```

   `annotations_trainval2017.zip` is already downloaded and md5-verified in
   `~/hafnia/public_dataset_downloads/coco-2017/.archives/`, so only `val2017.zip` (815 MB) is
   missing. The network on this machine ran at ~300 KB/s, so expect ~45 min.

2. **`ARCHIVE_IMAGES_VAL.md5 = "442b8da7639aecaf257c1dab9a6b9cc4"` is unverified.** The md5 of
   `annotations_trainval2017.zip` and `image_info_test2017.zip` were confirmed from their S3 ETags,
   but `val2017.zip` is a multipart upload so its ETag is not an md5. The value above is the one
   commonly published for COCO val2017 — if step 1 fails with a checksum error, verify with
   `md5sum` and correct the constant (do not just drop the check).

3. A leftover partial download from a checksum experiment is in `~/hafnia/coco_downloads_check/`
   (~255 MB, incomplete). Safe to delete — I was denied permission to remove it.

## Next steps (the follow-up prompt)

- Enable the remaining splits by extending `SUPPORTED_SPLITS` in `format_coco_2017.py`. Things to
  settle when doing that:
  - **The test split has no public annotations.** `image_info_test2017.json` contains `images` and
    `categories` but no `annotations`, so the existing importer produces samples without primitives.
    `check_dataset_tasks` requires at least one annotation per task, so a train+val+test dataset will
    need either a special case for the test split or a documented "no annotations" behaviour.
  - `train2017.zip` is 19 GB (~118k images). The conversion loads
    `instances_train2017.json` (~450 MB) and `person_keypoints_train2017.json` fully into memory.
  - `n_samples` is split evenly across splits by `from_coco_dataset_by_split_definitions` and takes
    the **first** N images of each split, not a random sample.

## Known limitation worth a decision

COCO labels a varying number of the 17 body keypoints per person, but hafnia requires a `Skeleton` to
carry every keypoint of its class template (`check_dataset_skeletons`). All 17 keypoints are therefore
kept, with the COCO visibility in `KeyPoint.meta["visibility"]` (`0`=not labeled, `1`=labeled but not
visible, `2`=visible), and unlabeled keypoints keep COCO's `(0, 0)` coordinate.

Consequence: `sample.draw_annotations(tasks=...)` draws edges from the pose towards the top-left
corner for every unlabeled keypoint — visible in the expected image
`tests/data/expected_images/test_format_coco_2017/test_from_coco_2017_keypoints_visualized.png`.
The data is faithful and trainable (mask on `visibility == 0`), but the visualization is misleading.

Options, if you want it fixed:
1. Add a `visible`/`labeled` field to `KeyPoint` and skip those keypoints and their edges in
   `Skeleton.draw`. Cleanest, but changes the primitive schema.
2. Skip unlabeled keypoints in `Skeleton.draw` based on `meta["visibility"]` — keeps the schema, but
   puts COCO semantics into a generic primitive.
3. Leave as is and document it.

I did not pick one, as it changes the `Skeleton`/`KeyPoint` API beyond this task.
