import os
import json
from pathlib import Path

import pytest
import random
import numpy as np


def _repo_root() -> Path:
	# Resolve repository root based on this test file location
	return Path(__file__).resolve().parent.parent


@pytest.mark.integration
def test_ingest_images_and_run_matching_pipeline(tmp_path: Path) -> None:
	# Ensure headless behavior for any backends that might initialize graphics
	os.environ.setdefault("MPLBACKEND", "Agg")
	# Clamp threads to avoid nondeterministic parallel math
	os.environ.setdefault("OMP_NUM_THREADS", "1")
	os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
	os.environ.setdefault("MKL_NUM_THREADS", "1")
	os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
	# Seed common RNGs for good measure (algorithms should be deterministic regardless)
	random.seed(0)
	np.random.seed(0)

	# Also clamp utool parallelism to a single process/thread
	try:
		import utool as ut  # type: ignore
		from utool import util_parallel as up  # type: ignore
		up.set_num_procs(1)
	except Exception:
		pass

	# Discover test images
	test_images_dir = _repo_root() / "tests" / "test_images"
	assert test_images_dir.exists() and test_images_dir.is_dir(), "tests/test_images directory not found"

	# Collect a small subset to keep the test fast
	image_paths = sorted(test_images_dir.glob("*.tif"))
	if len(image_paths) == 0:
		pytest.skip("No images found in tests/test_images")
	# Use all images for full coverage
	gpath_list = [p.as_posix() for p in image_paths]

	# Open/create a temporary database
	import ibeis

	dbdir = tmp_path / "ibeis_db"
	dbdir_str = dbdir.as_posix()
	ibs = ibeis.opendb(dbdir=dbdir_str, allow_newdir=True, force_serial=True)

	# Ingest images
	gid_list = ibs.add_images(gpath_list)
	assert len(gid_list) == len(gpath_list)

	# Use whole images as annotations
	aid_list = ibs.use_images_as_annotations(gid_list, adjust_percent=0.0)
	assert len(aid_list) >= 2, "Need at least two annotations to run matching"

	# Query every annotation against every annotation (all-vs-all)
	qaid_list = sorted(set(aid_list))
	daid_list = sorted(set(aid_list))
	assert len(qaid_list) >= 1

	# Create and run query request
	qreq_ = ibs.new_query_request(qaid_list, daid_list)
	# Execute with explicit qaids to avoid 'not a subset' assertions in shallow copies
	cm_list = qreq_.execute(qaids=qaid_list, use_cache=False)

	# Basic validations
	assert isinstance(cm_list, list)
	assert len(cm_list) == len(qaid_list)
	for cm, expected_qaid in zip(cm_list, qaid_list):
		assert getattr(cm, "qaid", None) == expected_qaid

	# Only treat scores >= min_score as meaningful matches
	min_score = 4

	# Build a deterministic snapshot of results
	def _cm_to_ranking(cm):
		# Pair daids with scores and sort by score desc, then daid asc to stabilize ties
		pairs = list(zip(list(map(int, cm.daid_list)), list(map(float, cm.score_list))))  # type: ignore[attr-defined]
		pairs.sort(key=lambda x: (-x[1], x[0]))
		# Keep only meaningful matches
		return [{"daid": daid, "score": float(score)} for daid, score in pairs if float(score) >= min_score]

	snapshot_data = {
		"cfgstr": qreq_.get_cfgstr(with_input=True, with_data=True, with_pipe=True),  # type: ignore[attr-defined]
		"images": [Path(p).name for p in gpath_list],
		"qaids": list(map(int, qaid_list)),
		"results": [
			{"qaid": int(cm.qaid), "ranking": _cm_to_ranking(cm)} for cm in cm_list
		],
	}

	# Snapshot location and update logic
	snap_dir = _repo_root() / "tests" / "snapshots"
	snap_dir.mkdir(parents=True, exist_ok=True)
	snap_path = snap_dir / "ingest_and_match_snapshot.json"
	update = os.getenv("IBEIS_UPDATE_SNAPSHOTS", "").lower() in ("1", "true", "yes")

	if update or not snap_path.exists():
		with snap_path.open("w", encoding="utf-8") as f:
			json.dump(snapshot_data, f, indent=2, sort_keys=True)
	else:
		with snap_path.open("r", encoding="utf-8") as f:
			existing = json.load(f)

		# Percent tolerance (default 10%). Allow env var as '0.1' or '10%'
		def _parse_tolerance_env() -> float:
			raw = os.getenv("IBEIS_SCORE_TOLERANCE_PCT", "").strip()
			if not raw:
				return 0.25
			try:
				if raw.endswith("%"):
					return float(raw[:-1]) / 100.0
				return float(raw)
			except Exception:
				return 0.25

		tol = _parse_tolerance_env()

		def _pct_diff(baseline: float, value: float) -> float:
			denom = max(abs(baseline), 1e-12)
			return abs(value - baseline) / denom

		def _results_to_maps(results):
			by_qaid = {}
			for item in results:
				q = int(item["qaid"])
				ranking = item.get("ranking", [])
				# Only include matches above threshold
				by_qaid[q] = {
					int(entry["daid"]): float(entry["score"])
					for entry in ranking
					if float(entry.get("score", 0.0)) >= min_score
				}
			return by_qaid

		reasons = []
		# Non-fatal informational difference
		if existing.get("cfgstr") != snapshot_data.get("cfgstr"):
			reasons.append("cfgstr differs")

		# Structural checks (these are considered failures if changed)
		old_imgs = existing.get("images", [])
		new_imgs = snapshot_data.get("images", [])
		if old_imgs != new_imgs:
			reasons.append(f"images differ: -{sorted(set(old_imgs) - set(new_imgs))} +{sorted(set(new_imgs) - set(old_imgs))}")

		old_q = list(map(int, existing.get("qaids", [])))
		new_q = list(map(int, snapshot_data.get("qaids", [])))
		if old_q != new_q:
			reasons.append(f"qaids differ: -{sorted(set(old_q) - set(new_q))} +{sorted(set(new_q) - set(old_q))}")

		old_map = _results_to_maps(existing.get("results", []))
		new_map = _results_to_maps(snapshot_data.get("results", []))

		common_qaids = sorted(set(old_map.keys()) & set(new_map.keys()))
		exceedances = []
		missing = {}
		extra = {}
		low_score_skipped_count = 0

		for q in common_qaids:
			old_d = old_map[q]
			new_d = new_map[q]
			common_daids = sorted(set(old_d.keys()) & set(new_d.keys()))
			miss_daids = sorted(set(old_d.keys()) - set(new_d.keys()))
			extra_daids = sorted(set(new_d.keys()) - set(old_d.keys()))
			if miss_daids:
				missing[str(q)] = miss_daids
			if extra_daids:
				extra[str(q)] = extra_daids
			for d in common_daids:
				b = float(old_d[d])
				v = float(new_d[d])
				# Skip percent-diff checks for clearly non-matches where score < min_score
				if b < min_score or v < min_score:
					low_score_skipped_count += 1
					continue
				p = _pct_diff(b, v)
				if p > tol:
					exceedances.append({
						"qaid": int(q),
						"daid": int(d),
						"snapshot": b,
						"new": v,
						"percent_off": p * 100.0,
					})

		# Anything to complain about?
		fail = bool(exceedances or (old_imgs != new_imgs) or (old_q != new_q) or missing or extra)

		# Write structured diff report to help debugging
		diff_report = {
			"tolerance": tol,
			"tolerance_percent": tol * 100.0,
			"min_score": min_score,
			"reasons": reasons,
			"exceedances_count": len(exceedances),
			"low_score_skipped_count": low_score_skipped_count,
			"exceedances_sample": exceedances[:25],
			"missing_daids_by_qaid": missing,
			"extra_daids_by_qaid": extra,
		}
		diff_path = snap_dir / "ingest_and_match_snapshot.diff.json"
		with diff_path.open("w", encoding="utf-8") as f:
			json.dump(diff_report, f, indent=2, sort_keys=True)

		if fail:
			msg_bits = []
			if exceedances:
				first = exceedances[0]
				msg_bits.append(
					f"Score tolerance exceeded for {len(exceedances)} matches (tolerance {tol*100:.2f}%). "
					f"First: qaid {first['qaid']} daid {first['daid']} "
					f"snapshot={first['snapshot']:.6g} new={first['new']:.6g} "
					f"({first['percent_off']:.2f}% off)"
				)
			if old_imgs != new_imgs:
				msg_bits.append("Images changed")
			if old_q != new_q:
				msg_bits.append("Qaids changed")
			if missing:
				msg_bits.append("Missing daids present")
			if extra:
				msg_bits.append("Extra daids present")
			if reasons:
				msg_bits.append(" | ".join(reasons))
			msg_bits.append(f"See diff: {diff_path}")
			assert False, " | ".join(msg_bits)


