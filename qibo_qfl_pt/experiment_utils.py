"""Helpers to build structured paths and discover saved experiments."""

from pathlib import Path
from collections import defaultdict
import json
import glob


class ExperimentPath:
    """Builds and parses the structured paths of the experiments.

    Schema: results/{campaign}/federated/{distribution}/{strategy}/{mode}[/{noise}]/

    This is the same tree that thesis_plots reads (see thesis_plots/paths.py),
    so a run launched with the defaults lands where the figures look for it.
    The noise level is a folder only for noisy and mitigated runs; a noiseless
    run has none. The number of shots is not part of the path: it is recorded
    in the "info" block of every run JSON.
    """

    ROOT = "results"
    CAMPAIGN = "noise_and_mitigation"
    PIPELINE = "federated"

    @staticmethod
    def build(
        strategy: str,
        mode: str,
        noise: str | None = None,
        nshots: int | str | None = None,
        distribution: str = "iid",
        root: str | None = None,
        campaign: str | None = None,
    ) -> str:
        """Build the save_path from the experiment parameters.

        Args:
            strategy: e.g. "fedavg", "fedadam", "fedyogi"
            mode: "noiseless", "noisy", "mitigated"
            noise: e.g. "uniform_p0.005_r0.005", "scaled_p0.005_r0.005_s0.002";
                ignored when mode is "noiseless"
            nshots: accepted for compatibility, no longer part of the path
            distribution: "iid" or "non_iid"
            root: root directory (default: "results")
            campaign: campaign folder under root (default: "noise_and_mitigation")

        Returns:
            Path as a string, e.g.
            "results/noise_and_mitigation/federated/iid/fedavg/noisy/scaled_p0.005_r0.005_s0.002"
        """
        parts = [
            root or ExperimentPath.ROOT,
            campaign or ExperimentPath.CAMPAIGN,
            ExperimentPath.PIPELINE,
            distribution,
            strategy.lower(),
            mode,
        ]
        if mode != "noiseless":
            if not noise:
                raise ValueError(f"mode '{mode}' needs a noise label")
            parts.append(noise)
        return "/".join(parts)

    @staticmethod
    def parse(path: str) -> dict:
        """Extract the parameters from a structured path.

        Returns:
            dict with keys: campaign, distribution, strategy, mode, noise
            (None for noiseless runs), path. A path that does not follow the
            schema returns only "path".
        """
        parts = Path(path).parts
        # Anchor on the "federated" level and read the levels below it.
        for i, part in enumerate(parts):
            if part == ExperimentPath.PIPELINE and i >= 1 and len(parts) >= i + 4:
                mode = parts[i + 3]
                info = {
                    "campaign": parts[i - 1],
                    "distribution": parts[i + 1],
                    "strategy": parts[i + 2],
                    "mode": mode,
                    "noise": None,
                    "path": str(path),
                }
                if mode != "noiseless" and len(parts) > i + 4:
                    info["noise"] = parts[i + 4]
                return info
        return {"path": str(path)}

    @staticmethod
    def noise_label(base_pauli: float, base_readout: float, scale: float = 0) -> str:
        """Build the noise config string from the numeric parameters.

        Args:
            base_pauli: base Pauli error
            base_readout: base readout error
            scale: spread between clients (0 = uniform)

        Returns:
            e.g. "uniform_p0.005_r0.005" or "scaled_p0.005_r0.005_s0.002"
        """
        if scale == 0:
            return f"uniform_p{base_pauli}_r{base_readout}"
        return f"scaled_p{base_pauli}_r{base_readout}_s{scale}"


def discover_experiments(root: str | None = None) -> list[dict]:
    """Scan the results directory and return the list of experiments found.

    An experiment is a folder that holds run JSONs (*_seed*.json) and follows
    the ExperimentPath schema. Every element is a dict with: campaign,
    distribution, strategy, mode, noise, nshots, n_seeds, n_files, path.
    nshots is read from the "info" block of the first run, when present.

    With pandas available, convert with:
        pd.DataFrame(discover_experiments())
    """
    root = root or ExperimentPath.ROOT
    experiments = []

    folders = sorted({f.parent for f in Path(root).rglob("*_seed*.json")})
    for folder in folders:
        info = ExperimentPath.parse(str(folder))
        if "distribution" not in info:
            continue

        json_files = sorted(folder.glob("*.json"))
        seed_files = [f for f in json_files if "_seed" in f.name]
        info["n_seeds"] = len(seed_files)
        info["n_files"] = len(json_files)
        info["nshots"] = None
        try:
            with open(seed_files[0], encoding="utf-8") as fh:
                info["nshots"] = json.load(fh).get("info", {}).get("nshots")
        except (OSError, ValueError):
            pass
        experiments.append(info)

    # Sort, so the output stays consistent
    experiments.sort(key=lambda x: x["path"])
    return experiments
