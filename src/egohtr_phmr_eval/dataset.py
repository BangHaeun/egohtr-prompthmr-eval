from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path

import numpy as np


def ensure_sequence_files(dataset_root: Path, sequence: str, repo_id: str | None) -> tuple[Path, Path]:
    seq_dir = dataset_root / "sequences" / sequence
    archive = seq_dir / "full_sequence.npz"
    config = seq_dir / "config.yaml"
    missing = [p.name for p in (archive, config) if not p.is_file()]
    if missing:
        if not repo_id:
            raise FileNotFoundError(
                f"Missing {', '.join(missing)} in {seq_dir}. Set huggingface_repo_id and log in with `hf auth login`, "
                "or place only full_sequence.npz and config.yaml there."
            )
        from huggingface_hub import hf_hub_download

        seq_dir.mkdir(parents=True, exist_ok=True)
        for filename in missing:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"sequences/{sequence}/{filename}",
                local_dir=str(dataset_root),
            )
    return archive, config


def list_npz_keys(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return {Path(n).stem for n in zf.namelist() if n.endswith(".npy")}


def _read_exact(stream, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = stream.read(min(8 * 1024 * 1024, count - len(chunks)))
        if not chunk:
            raise EOFError("Unexpected end of npz array")
        chunks.extend(chunk)
    return bytes(chunks)


def _skip(stream, count: int) -> None:
    while count:
        block = stream.read(min(8 * 1024 * 1024, count))
        if not block:
            raise EOFError("Unexpected end while seeking to RGB frame")
        count -= len(block)


def read_npz_rgb_frame(path: Path, key: str, index: int) -> np.ndarray:
    """Read one frame from a compressed .npz member without materializing the whole video array."""
    with zipfile.ZipFile(path) as zf:
        member = f"{key}.npy"
        if member not in zf.namelist():
            raise KeyError(f"{key!r} is not present in {path}")
        with zf.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in ((2, 0), (3, 0)):
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            else:
                raise ValueError(f"Unsupported NPY format {version}")
            if len(shape) < 2 or not 0 <= index < shape[0]:
                raise IndexError(f"Frame index {index} out of range for shape {shape}")
            if fortran:
                raise ValueError("Fortran-ordered RGB arrays are not supported by incremental frame reading")
            frame_shape = shape[1:]
            frame_bytes = int(np.prod(frame_shape)) * dtype.itemsize
            _skip(stream, frame_bytes * index)
            frame = np.frombuffer(_read_exact(stream, frame_bytes), dtype=dtype).reshape(frame_shape)
            return frame


def select_index(npz_path: Path, index: int | None, timestamp_seconds: float | None) -> tuple[int, float]:
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        ts_key = "aria_timestamp" if "aria_timestamp" in keys else "exomc_timestamp"
        timestamps = np.asarray(data[ts_key], dtype=np.float64)
        if index is None:
            if timestamp_seconds is None:
                raise ValueError("Set either index or timestamp_seconds")
            idx = int(np.argmin(np.abs(timestamps - timestamp_seconds)))
        else:
            idx = int(index)
        if not 0 <= idx < len(timestamps):
            raise IndexError(f"index {idx} outside {ts_key} length {len(timestamps)}")
        t = float(timestamps[idx])
        if "aria_timestamp" in keys and "exomc_timestamp" in keys:
            aria_t = float(data["aria_timestamp"][idx])
            exo_t = float(data["exomc_timestamp"][idx])
            if abs(aria_t - exo_t) > 1e-5:
                raise ValueError(f"Unsynchronized timestamps at {idx}: Aria={aria_t}, ExoMC={exo_t}")
            t = aria_t
        return idx, t
