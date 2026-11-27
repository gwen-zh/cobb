"""Verify DMS variants and extract wild-type residue features from the reference structures."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Data.IUPACData import protein_weights
from Bio.PDB import DSSP, PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.SeqUtils.ProtParamData import kd
from scipy.spatial.distance import cdist

from settings import AA, CACHE, DATA, RSA_CUTOFF, save_json, selected_assays


MAX_VARIANTS = 1500
SAMPLE_SEED = 2035
CONFIDENCE_CUTOFF = 70.0
VALID_MUTATION = re.compile(f"([{AA}])([1-9][0-9]*)([{AA}])")
BLOSUM62 = substitution_matrices.load("BLOSUM62")
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}


def standard_residues(chain):
    return [residue for residue in chain if residue.id[0] == " "
            and residue.resname in protein_letters_3to1 and "CA" in residue]


class ProteinChain(Select):
    def __init__(self, chain_id, residues):
        self.chain_id = chain_id
        self.residues = {residue.id for residue in residues}

    def accept_model(self, model):
        return model.id == 0

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id in self.residues


def sequence_position_map(target: str, observed: str) -> dict[int, int]:
    if observed == target:
        return {idx: idx for idx in range(len(target))}
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-3,
                              open_gap_score=-8, extend_gap_score=-0.5)
    alignment = aligner.align(target, observed)[0]
    mapping = {}
    for target_span, observed_span in zip(*alignment.aligned, strict=True):
        for i, j in zip(range(*target_span), range(*observed_span), strict=True):
            if target[i] == observed[j]:
                mapping[i] = j
    return mapping


def structure_features(path: Path, target: str, folder: Path, foldseek: str, dssp_bin: str) -> tuple[pd.DataFrame, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("reference", path)
    candidates = []
    for chain in structure[0]:
        residues = standard_residues(chain)
        sequence = "".join(protein_letters_3to1[residue.resname] for residue in residues)
        if sequence:
            mapping = sequence_position_map(target, sequence)
            candidates.append((len(mapping), chain.id, residues, sequence, mapping))
    if not candidates:
        raise ValueError(f"No protein chain in {path}")
    matches, chain_id, residues, sequence, mapping = max(candidates, key=lambda item: item[0])
    if matches / len(target) < 0.95:
        raise ValueError(f"Only {matches}/{len(target)} positions match {path.name}")

    folder.mkdir(parents=True, exist_ok=True)
    normalized = folder / "reference.pdb"
    with normalized.open("w") as handle:
        handle.write("HEADER    PROTEINGYM ALPHAFOLD2 REFERENCE\n")
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(handle, ProteinChain(chain_id, residues))
    model = parser.get_structure("normalized", normalized)[0]

    with tempfile.TemporaryDirectory(prefix="dssp-foldseek-") as tmp:
        descriptor_path = Path(tmp) / "descriptors.tsv"
        dssp_path = Path(tmp) / "annotation.dssp"
        for command in ([dssp_bin, "--output-format", "dssp", str(normalized), str(dssp_path)],
                        [foldseek, "structureto3didescriptor", "-v", "0", "--threads", "1",
                         "--chain-name-mode", "1", str(normalized), str(descriptor_path)]):
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"{command[0]} failed for {path.name}: {completed.stderr or completed.stdout}")
        dssp = DSSP(model, str(dssp_path), file_type="DSSP", acc_array="Wilke")
        lines = descriptor_path.read_text().splitlines()
        if len(lines) != 1:
            raise ValueError(f"Expected one Foldseek chain for {path.name}, found {len(lines)}")
        _, foldseek_seq, foldseek_structure = lines[0].split("\t")[:3]
    if foldseek_seq != sequence or len(foldseek_structure) != len(sequence):
        raise ValueError(f"Foldseek coordinates and PDB residues do not map exactly in {path.name}")

    xyz = np.array([residue["CA"].coord for residue in residues])
    dist = cdist(xyz, xyz)
    nearby = np.abs(np.arange(len(residues))[:, None] - np.arange(len(residues))[None, :]) <= 2
    contact = np.count_nonzero((dist < 8.0) & ~nearby, axis=1)
    three_di = ["#"] * len(target)
    records = []
    for seq_idx, struct_idx in mapping.items():
        residue = residues[struct_idx]
        ca = residue["CA"]
        pl_ddt = ca.bfactor
        token = foldseek_structure[struct_idx].upper()
        three_di[seq_idx] = token.lower() if token in AA and pl_ddt >= CONFIDENCE_CUTOFF else "#"
        key = (chain_id, residue.id)
        if key not in dssp:
            continue
        _, aa, ss8, rsa, *_ = dssp[key]
        if aa.upper() != target[seq_idx]:
            raise ValueError(f"DSSP amino acid mismatch at {path.name} position {seq_idx + 1}")
        if not isinstance(rsa, (int, float)) or not np.isfinite(rsa):
            continue
        ss3 = "helix" if ss8 in "HGI" else "strand" if ss8 in "EB" else "coil"
        records.append({"position": seq_idx + 1, "wt": target[seq_idx], "chain": chain_id,
                        "pdb_resi": f"{residue.id[1]}{residue.id[2].strip()}",
                        "rsa": float(rsa), "dssp8": ss8, "ss3": ss3, "contacts": int(contact[struct_idx]),
                        "plddt": float(pl_ddt), "a3di": three_di[seq_idx],
                        "environment": "buried" if rsa < RSA_CUTOFF else "exposed",
                        "x": float(ca.coord[0]), "y": float(ca.coord[1]), "z": float(ca.coord[2])})
    return pd.DataFrame(records).sort_values("position"), "".join(three_di)


def load_variants(path: Path, sequence: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, usecols=["mutant", "mutated_sequence", "DMS_score"])
    records = []
    for mutant, sequence_observed, score in raw.itertuples(index=False, name=None):
        match = VALID_MUTATION.fullmatch(str(mutant))
        if match is None or not np.isfinite(score):
            continue
        wt, pos, alt = match.groups()
        position = int(pos)
        if wt == alt:
            continue
        i = position - 1
        if i >= len(sequence) or sequence[i] != wt:
            raise ValueError(f"Incorrect WT residue for {mutant} in {path.name}")
        if sequence[:i] + alt + sequence[i + 1:] != sequence_observed:
            raise ValueError(f"Variant sequence does not match {mutant} in {path.name}")
        records.append((mutant, position, wt, alt, float(score)))
    variants = pd.DataFrame(records, columns=["mutant", "position", "wt", "alt", "score"])
    if variants["mutant"].duplicated().any():
        raise ValueError(f"Duplicate single substitutions in {path.name}")
    return variants, {"raw_rows": len(raw), "finite_singles": len(variants)}


def add_descriptors(variants: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for letter in AA:
        columns[f"d_wt_{letter}"] = (variants["wt"] == letter).astype(int)
        columns[f"d_alt_{letter}"] = (variants["alt"] == letter).astype(int)
    pair = list(zip(variants["wt"], variants["alt"], strict=True))
    columns["d_blosum62"] = [float(BLOSUM62[a, b]) for a, b in pair]
    columns["d_hydropathy_change"] = [kd[b] - kd[a] for a, b in pair]
    columns["d_mass_change"] = [protein_weights[b] - protein_weights[a] for a, b in pair]
    columns["d_charge_change"] = [CHARGE.get(b, 0) - CHARGE.get(a, 0) for a, b in pair]
    columns["s_rsa"] = variants["rsa"]
    columns["s_contacts"] = variants["contacts"]
    for ss in ("helix", "strand", "coil"):
        columns[f"s_{ss}"] = (variants["ss3"] == ss).astype(int)
    return pd.concat([variants, pd.DataFrame(columns, index=variants.index)], axis=1)


def prepare(assay: dict, reference: dict, foldseek: str, dssp_bin: str) -> dict:
    assay_id = assay["id"]
    source = DATA / "raw"
    folder = DATA / "processed" / assay_id
    info = reference[assay_id]
    target = info["target_seq"]
    if set(target) - set(AA):
        raise ValueError(f"Noncanonical target sequence in {assay_id}")
    residues, three_di = structure_features(source / "structures" / info["pdb_file"], target, folder, foldseek, dssp_bin)
    variants, counts = load_variants(source / "assays" / info["DMS_filename"], target)
    mapped = variants.merge(residues, on=["position", "wt"], how="inner", validate="many_to_one")
    if len(mapped) < 1000:
        raise ValueError(f"Only {len(mapped)} mapped singles for {assay_id}")
    mapped = mapped.sort_values(["position", "alt"]).reset_index(drop=True)
    if len(mapped) > MAX_VARIANTS:
        # Sample without using scores and restore the position/allele order afterward.
        ids = np.random.default_rng(SAMPLE_SEED).choice(len(mapped), MAX_VARIANTS, replace=False)
        sample = mapped.iloc[np.sort(ids)].reset_index(drop=True)
    else:
        sample = mapped
    sample = add_descriptors(sample)
    sample.insert(0, "variant_id", assay_id + ":" + sample["mutant"])
    sample.to_csv(folder / "variants.csv", index=False)
    residues.to_csv(folder / "residues.csv", index=False)
    metadata = {**assay, "reference_pdb": info["pdb_file"], "sequence": target, "three_di": three_di,
                **counts, "mapped_singles": len(mapped), "selected_singles": len(sample),
                "mutated_positions": int(sample["position"].nunique()), "structure_positions": len(residues),
                "structure_length": len(target), "mask_count": three_di.count("#"),
                "buried_singles": int((sample["rsa"] < RSA_CUTOFF).sum()),
                "exposed_singles": int((sample["rsa"] >= RSA_CUTOFF).sum())}
    save_json(folder / "metadata.json", metadata)
    print(f"{assay['label']}: {len(variants):,} singles, {len(mapped):,} mapped, {len(sample):,} selected, "
          f"{metadata['mutated_positions']} sites, {metadata['mask_count']} masked 3Di tokens", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assay", nargs="+")
    parser.add_argument("--foldseek", default=str(CACHE / "bin" / "foldseek"))
    parser.add_argument("--dssp", default="mkdssp")
    args = parser.parse_args()
    assays = selected_assays(args.assay)
    reference = json.loads((DATA / "raw" / "reference.json").read_text())
    for assay in assays:
        prepare(assay, reference, args.foldseek, args.dssp)


if __name__ == "__main__":
    main()
