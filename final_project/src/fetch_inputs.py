"""Fetch pinned public inputs and retain only the four selected benchmark assays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from settings import ASSAYS, CACHE, DATA, MODEL_SPECS, save_json


RELEASE = "https://zenodo.org/records/15293562/files"
REFERENCE = ("https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
             "1f8de974dead8ff7501eff087b725d14a965e9f9/reference_files/DMS_substitutions.csv")
FOLDSEEK = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
ARCHIVES = {
    "DMS_ProteinGym_substitutions.zip": "ca1a4d46941ef33cc972245347118c7d",
    "ProteinGym_AF2_structures.zip": "442881616899de02cd1c1b0a57badd37",
}
MODEL_FILES = ("config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt")


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def fetch(url: str, path: Path, md5: str | None = None) -> Path:
    if path.is_file() and path.stat().st_size and (md5 is None or digest(path, "md5") == md5):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request_url = url
            if "huggingface.co/" in url:
                request_url = f"{url}?download=true&request={time.time_ns()}"
            with requests.get(request_url, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                with staging.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
            if md5 and digest(staging, "md5") != md5:
                raise ValueError(f"Official MD5 verification failed: {path.name}")
            staging.replace(path)
            print(f"Fetched {path.name} ({path.stat().st_size:,} bytes)", flush=True)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"No response for {url}")


def extract_named(archive: Path, names: set[str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        mapping = {Path(member.filename).name: member for member in bundle.infolist()
                   if not member.is_dir() and "__MACOSX" not in member.filename}
        if missing := names - mapping.keys():
            raise ValueError(f"{archive.name} is missing {sorted(missing)}")
        for name in sorted(names):
            staging = out / f"{name}.tmp"
            with bundle.open(mapping[name]) as source, staging.open("wb") as target:
                shutil.copyfileobj(source, target)
            staging.replace(out / name)
    return {name: digest(out / name) for name in sorted(names)}


def data_inputs() -> None:
    source = (DATA / "raw")
    source.mkdir(parents=True, exist_ok=True)
    reference = requests.get(REFERENCE, timeout=(15, 30))
    reference.raise_for_status()
    assay_ids = {assay["id"] for assay in ASSAYS}
    rows = {row["DMS_id"]: row for row in csv.DictReader(io.StringIO(reference.text))
            if row["DMS_id"] in assay_ids}
    if set(rows) != assay_ids:
        raise ValueError(f"Missing reference rows: {assay_ids - rows.keys()}")
    save_json(source / "reference.json", rows)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(fetch, f"{RELEASE}/{name}", CACHE / name, md5)
                   for name, md5 in ARCHIVES.items()}
        assay_zip, structure_zip = (future.result() for future in futures.values())
    selected_csv = extract_named(assay_zip, {row["DMS_filename"] for row in rows.values()}, source / "assays")
    selected_pdb = extract_named(structure_zip, {row["pdb_file"] for row in rows.values()}, source / "structures")
    save_json(source / "provenance.json", {"release": RELEASE, "reference": REFERENCE,
              "archive_md5": ARCHIVES, "assay_sha256": selected_csv, "structure_sha256": selected_pdb})


def model_inputs(key: str) -> None:
    name, revision, weights = MODEL_SPECS[key]
    folder = CACHE / "models" / key
    for filename in (*MODEL_FILES, weights):
        fetch(f"https://huggingface.co/{name}/resolve/{revision}/{filename}", folder / filename)
    save_json(folder / "provenance.json", {"repo": name, "revision": revision,
              "sha256": {filename: digest(folder / filename) for filename in (*MODEL_FILES, weights)}})


def foldseek_binary() -> None:
    archive = fetch(FOLDSEEK, CACHE / "foldseek-linux-avx2.tar.gz")
    dest = CACHE / "bin" / "foldseek"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        member, = (member for member in bundle.getmembers() if member.isfile()
                   and member.name.endswith("/bin/foldseek"))
        with bundle.extractfile(member) as source, dest.open("wb") as target:
            shutil.copyfileobj(source, target)
    dest.chmod(0o755)
    save_json(dest.parent / "foldseek.json", {"source": FOLDSEEK, "sha256": digest(dest)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", nargs="+", choices=("data", "foldseek", *MODEL_SPECS),
                        default=("data", "foldseek", *MODEL_SPECS))
    args = parser.parse_args()
    handlers = {"data": data_inputs, "foldseek": foldseek_binary}
    with ThreadPoolExecutor(max_workers=len(args.part)) as executor:
        futures = [executor.submit(handlers[key]) if key in handlers else executor.submit(model_inputs, key)
                   for key in args.part]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
