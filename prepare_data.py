"""Restore the release's split gzip streams into their original NPZ files."""
import argparse
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zlib


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PARTS = DATA / "_parts"
BLOCK = 1024 * 1024


@contextmanager
def preparation_lock():
    # The OS releases this lock when a process exits, including after interruption.
    key = hashlib.sha256(os.fsencode(os.path.normcase(str(ROOT)))).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"prepare-data-{key}.lock"
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def indexed_path(name, directory):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(directory) or path == directory:
        raise ValueError(f"Index path outside {directory.name}: {name}")
    return path


def existing_file(path, size, checksum, verify):
    if not path.exists():
        return False
    if path.stat().st_size != size:
        raise ValueError(f"Existing file has the wrong size: {path.relative_to(ROOT)}")
    if verify:
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != checksum:
                raise ValueError(f"Existing file has the wrong SHA-256: {path.relative_to(ROOT)}")
    return True


def unpack(entry, verify):
    target = indexed_path(entry["path"], DATA)
    if target.suffix != ".npz" or target.is_relative_to(PARTS):
        raise ValueError(f"Unexpected dataset destination: {entry['path']}")
    size, checksum = entry["bytes"], entry["sha256"]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Only the lock holder uses this name. A retry removes any interrupted write.
    pending = target.with_name(f".{target.name}.prepare-data.tmp")
    pending.unlink(missing_ok=True)
    if existing_file(target, size, checksum, verify):
        return False

    digest = hashlib.sha256()
    length = 0
    stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        with pending.open("xb") as dest:
            for name in entry["parts"]:
                with indexed_path(name, PARTS).open("rb") as part:
                    while chunk := part.read(64 * 1024):
                        while chunk:
                            block = stream.decompress(chunk, BLOCK)
                            if stream.unused_data:
                                raise ValueError(f"Unexpected data after gzip stream: {name}")
                            length += len(block)
                            if length > size:
                                raise ValueError(f"Expanded file exceeds expected size: {target.name}")
                            dest.write(block)
                            digest.update(block)
                            chunk = stream.unconsumed_tail
        if not stream.eof or length != size or digest.hexdigest() != checksum:
            raise ValueError(f"Incomplete or damaged data for {target.relative_to(ROOT)}")
        if existing_file(target, size, checksum, True):
            return False
        os.replace(pending, target)
        return True
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="also check SHA-256 for previously restored files")
    args = parser.parse_args()
    with preparation_lock():
        entries = json.loads((PARTS / "index.json").read_text(encoding="utf-8-sig"))["files"]
        restored = sum(unpack(entry, args.verify) for entry in entries)
    print(f"Dataset files ready ({len(entries)} indexed, {restored} restored).")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, zlib.error) as error:
        sys.exit(f"Data preparation failed: {error}")
