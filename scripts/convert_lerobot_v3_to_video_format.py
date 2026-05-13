"""
Convert a LeRobot v3 dataset to mimic-video format.

LeRobot v3 packs multiple episodes into each video file, and the data parquet
chunks don't necessarily align with the video chunks. We therefore use the
per-episode metadata in meta/episodes/**/*.parquet, which records exactly
which video file each episode lives in and the from/to timestamps within it.

Extraction is timestamp-based via ffmpeg's trim filter, which avoids frame-count
mismatches that can arise with AV1 decoders.

Input:
  <lerobot_dir>/videos/<video_key>/chunk-<N>/file-<M>.mp4
  <lerobot_dir>/meta/episodes/chunk-<N>/file-<M>.parquet

Output:
  <output_dir>/video/ep_<NNN>.mp4
  <output_dir>/metas/ep_<NNN>.txt

Usage:
  pip install pyarrow tqdm
  python scripts/convert_lerobot_v3_to_video_format.py \
    --lerobot-dir ./data/push-that-thing \
    --output-dir  ./data/push-mimic \
    --ffmpeg      /usr/local/bin/ffmpeg
"""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lerobot-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--video-key", default="observation.images.front")
    p.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg binary (must support AV1)")
    return p.parse_args()


def extract_all_from_file(
    ffmpeg: str,
    src: Path,
    episodes: list[dict],  # each has: ep_idx, start_ts, end_ts, dst_video
) -> None:
    """Decode src once and write all episodes in a single ffmpeg call."""
    n = len(episodes)

    split = f"[0:v]split={n}" + "".join(f"[v{i}]" for i in range(n))

    trims = [
        f"[v{i}]trim=start={ep['start_ts']:.6f}:end={ep['end_ts']:.6f},setpts=PTS-STARTPTS[out{i}]"
        for i, ep in enumerate(episodes)
    ]

    filter_complex = split + ";" + ";".join(trims)

    cmd = [ffmpeg, "-y", "-i", str(src), "-filter_complex", filter_complex]
    for i, ep in enumerate(episodes):
        cmd += ["-map", f"[out{i}]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(ep["dst_video"])]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode())


def main() -> None:
    args = parse_args()
    (args.output_dir / "video").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metas").mkdir(parents=True, exist_ok=True)

    chunk_col = f"videos/{args.video_key}/chunk_index"
    file_col = f"videos/{args.video_key}/file_index"
    from_col = f"videos/{args.video_key}/from_timestamp"
    to_col = f"videos/{args.video_key}/to_timestamp"
    cols = ["episode_index", "tasks", chunk_col, file_col, from_col, to_col]

    video_root = args.lerobot_dir / "videos" / args.video_key
    by_file: dict[Path, list[dict]] = defaultdict(list)

    for parquet_path in sorted((args.lerobot_dir / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(parquet_path, columns=cols)
        for i in range(table.num_rows):
            ep_idx = table["episode_index"][i].as_py()
            task = table["tasks"][i].as_py()[0]
            chunk = table[chunk_col][i].as_py()
            file_idx = table[file_col][i].as_py()
            start_ts = table[from_col][i].as_py()
            end_ts = table[to_col][i].as_py()

            video_file = video_root / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
            by_file[video_file].append({
                "ep_idx": ep_idx,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "task": task,
                "dst_video": args.output_dir / "video" / f"ep_{ep_idx:03d}.mp4",
                "dst_meta": args.output_dir / "metas" / f"ep_{ep_idx:03d}.txt",
            })

    total_episodes = sum(len(eps) for eps in by_file.values())

    with tqdm(total=total_episodes, unit="ep") as bar:
        for video_file, episodes in sorted(by_file.items()):
            episodes.sort(key=lambda e: e["start_ts"])

            missing = [ep for ep in episodes if not ep["dst_video"].exists()]
            if not missing:
                bar.update(len(episodes))
                continue

            bar.set_description(video_file.name)
            extract_all_from_file(args.ffmpeg, video_file, missing)

            for ep in missing:
                ep["dst_meta"].write_text(ep["task"])

            bar.update(len(episodes))

    print(f"Done -> {args.output_dir}")


if __name__ == "__main__":
    main()
