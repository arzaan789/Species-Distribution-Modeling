"""Faithful boundary and preflight gate for the official DeepMaxent code."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from provenance_sdm.config import StudyConfig


@dataclass(frozen=True)
class GateReport:
    official_commit: str
    formula_check_passed: bool
    repository_example_passed: bool
    multi_seed_pilot_passed: bool
    projected_calendar_days: float
    comparable_predictions_passed: bool
    include: bool
    reasons: tuple[str, ...]


def normalized_poisson_reference_loss(
    counts: np.ndarray,
    logits: np.ndarray,
) -> float:
    """Return count-weighted negative log site probability per record."""

    target = np.asarray(counts, dtype=float)
    intensity = np.asarray(logits, dtype=float)
    if target.ndim != 2 or target.shape != intensity.shape:
        raise ValueError("count and logit tensors must have the same two-dimensional shape")
    if not np.isfinite(target).all() or not np.isfinite(intensity).all():
        raise ValueError("count and logit tensors must be finite")
    if np.any(target < 0):
        raise ValueError("counts must be non-negative")
    species_counts = target.sum(axis=0)
    if np.any(species_counts <= 0):
        raise ValueError("every species must have a positive count")
    log_probability = intensity - logsumexp(intensity, axis=0, keepdims=True)
    return float(-np.sum(target * log_probability) / target.sum())


class DeepMaxentAdapter:
    """Load model and loss classes from an official checkout at runtime."""

    def __init__(self, checkout: Path) -> None:
        self.checkout = Path(checkout)

    @property
    def commit(self) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
        commit = completed.stdout.strip()
        return commit if len(commit) == 40 else "unavailable"

    def _load_module(self, relative_path: str, name: str) -> ModuleType:
        path = self.checkout / relative_path
        if not path.is_file():
            raise ValueError(f"official DeepMaxent file is missing: {relative_path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import official DeepMaxent file: {relative_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def formula_check(self) -> tuple[bool, str | None]:
        """Compare the official loss with an independent NumPy oracle."""

        try:
            import torch

            losses = self._load_module(
                "librairies/losses.py",
                "_official_deepmaxent_losses",
            )
            counts = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)
            logits = np.log(np.array([[0.8, 0.25], [0.2, 0.75]], dtype=float))
            official_raw = losses.deepmaxent_loss()(
                torch.tensor(logits, dtype=torch.float64),
                torch.tensor(counts, dtype=torch.float64),
            )
            official_normalized = (
                float(official_raw.item()) * counts.size / counts.sum()
            )
            expected = normalized_poisson_reference_loss(counts, logits)
        except Exception as exc:
            return False, f"official formula check could not run: {type(exc).__name__}: {exc}"
        if not np.isclose(official_normalized, expected, rtol=1e-10, atol=1e-12):
            return False, "official loss does not match the normalized Poisson oracle"
        return True, None

    def repository_smoke_check(self) -> tuple[bool, str | None]:
        """Exercise official model/loss forward and backward operations."""

        try:
            import torch

            torch.manual_seed(17)
            models = self._load_module(
                "librairies/model.py",
                "_official_deepmaxent_model",
            )
            losses = self._load_module(
                "librairies/losses.py",
                "_official_deepmaxent_losses_smoke",
            )
            model = models.deepmaxent_model(3, 8, 2, 1).to(dtype=torch.float64)
            features = torch.linspace(
                -1,
                1,
                steps=36,
                dtype=torch.float64,
            ).reshape(12, 3)
            counts = torch.zeros((12, 2), dtype=torch.float64)
            counts[[1, 5, 9], 0] = torch.tensor(
                [1.0, 2.0, 1.0],
                dtype=torch.float64,
            )
            counts[[2, 6, 10], 1] = 1.0
            output = model(features)
            loss = losses.deepmaxent_loss()(output, counts)
            loss.backward()
            predictions = output.softmax(dim=0).detach().numpy()
            gradients = [
                parameter.grad.detach().numpy()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
        except Exception as exc:
            return False, f"official repository smoke check failed: {type(exc).__name__}: {exc}"
        if (
            not np.isfinite(predictions).all()
            or not np.allclose(predictions.sum(axis=0), 1.0)
            or not gradients
            or not all(np.isfinite(gradient).all() for gradient in gradients)
        ):
            return False, "official repository smoke check produced invalid values"
        return True, None

    def repository_tutorial_check(self) -> tuple[bool, str | None]:
        """Execute the reviewed core cells of the distributed tutorial."""

        notebook_path = self.checkout / "tutorial_deepmaxent.ipynb"
        biodiversity_path = self.checkout / "data/tutorial/biodiversity_data.csv"
        raster_dir = self.checkout / "data/tutorial/cropped_rasters"
        if (
            not notebook_path.is_file()
            or not biodiversity_path.is_file()
            or len(list(raster_dir.glob("*.tif"))) != 19
        ):
            return False, "official tutorial inputs are incomplete"
        prior_directory = Path.cwd()
        inserted_path = str(self.checkout)
        try:
            import rasterio
            import torch
            from sklearn.preprocessing import StandardScaler
            from tqdm import tqdm

            models = self._load_module(
                "librairies/model.py",
                "_official_deepmaxent_tutorial_model",
            )
            losses = self._load_module(
                "librairies/losses.py",
                "_official_deepmaxent_tutorial_losses",
            )
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            namespace = {
                "__builtins__": __builtins__,
                "np": np,
                "pd": pd,
                "os": os,
                "rasterio": rasterio,
                "torch": torch,
                "StandardScaler": StandardScaler,
                "tqdm": tqdm,
                "deepmaxent_model": models.deepmaxent_model,
                "deepmaxent_loss": losses.deepmaxent_loss,
            }
            reviewed_cells = (4, 6, 7, 23, 25, 27, 35, 38, 40, 42)
            sys.path.insert(0, inserted_path)
            os.chdir(self.checkout)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for cell_index in reviewed_cells:
                    source = "".join(notebook["cells"][cell_index]["source"])
                    exec(
                        compile(
                            source,
                            f"official_tutorial_cell_{cell_index}",
                            "exec",
                        ),
                        namespace,
                    )
                namespace["args"].epoch = 2
                training_source = "".join(notebook["cells"][43]["source"])
                exec(
                    compile(training_source, "official_tutorial_cell_43", "exec"),
                    namespace,
                )
            model = namespace["results"]["model"]
            validation = namespace["X_val_tensor"]
            device = namespace["device"]
            with torch.no_grad():
                probabilities = torch.softmax(
                    model(validation.to(device)).cpu(),
                    dim=0,
                ).numpy()
            best_loss = float(namespace["results"]["best_val_loss"])
        except Exception as exc:
            return False, f"official tutorial core failed: {type(exc).__name__}: {exc}"
        finally:
            os.chdir(prior_directory)
            if inserted_path in sys.path:
                sys.path.remove(inserted_path)
        if (
            not np.isfinite(best_loss)
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-5)
        ):
            return False, "official tutorial core produced invalid predictions"
        return True, None

    def fit_pilot(
        self,
        features: np.ndarray,
        counts: np.ndarray,
        seed: int,
        epochs: int,
    ) -> tuple[np.ndarray, float, float]:
        """Fit a pilot using official model/loss classes without copying them."""

        import torch

        torch.manual_seed(seed)
        models = self._load_module(
            "librairies/model.py",
            f"_official_deepmaxent_model_{seed}",
        )
        losses = self._load_module(
            "librairies/losses.py",
            f"_official_deepmaxent_losses_{seed}",
        )
        x = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(counts, dtype=torch.float32)
        model = models.deepmaxent_model(
            x.shape[1],
            250,
            target.shape[1],
            2,
        )
        criterion = losses.deepmaxent_loss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.00002,
            weight_decay=3e-4,
        )
        with torch.no_grad():
            initial = float(criterion(model(x), target).item())
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            output = model(x)
            final = float(criterion(output, target).item())
            prediction = output.softmax(dim=0).numpy()
        return prediction, initial, final


def run_deepmaxent_pilot(
    checkout: Path,
    output: Path,
    *,
    seeds: tuple[int, ...] = (17, 29, 43),
    n_sites: int = 4_096,
    n_species: int = 20,
    epochs: int = 20,
    full_site_count: int = 243_541,
    full_epochs: int = 100,
) -> Path:
    """Run a representative multi-seed official-code timing pilot."""

    if len(set(seeds)) < 3:
        raise ValueError("pilot requires at least three unique seeds")
    if min(n_sites, n_species, epochs, full_site_count, full_epochs) <= 0:
        raise ValueError("pilot dimensions and epochs must be positive")
    generator = np.random.default_rng(8_271)
    features = generator.normal(size=(n_sites, 9))
    coefficients = generator.normal(scale=0.7, size=(9, n_species))
    truth_logits = features @ coefficients
    truth_probability = np.exp(
        truth_logits - logsumexp(truth_logits, axis=0, keepdims=True)
    )
    counts = np.column_stack(
        [
            generator.multinomial(50, truth_probability[:, species_index])
            for species_index in range(n_species)
        ]
    ).astype(float)
    adapter = DeepMaxentAdapter(checkout)
    rows = []
    for seed in seeds:
        started = time.perf_counter()
        try:
            prediction, initial_loss, final_loss = adapter.fit_pilot(
                features,
                counts,
                seed,
                epochs,
            )
            comparable = (
                prediction.shape == counts.shape
                and np.isfinite(prediction).all()
                and np.all(prediction >= 0)
                and np.allclose(prediction.sum(axis=0), 1.0, atol=1e-6)
            )
            passed = (
                comparable
                and np.isfinite(initial_loss)
                and np.isfinite(final_loss)
                and final_loss < initial_loss
            )
            error = None
        except Exception as exc:
            initial_loss = np.nan
            final_loss = np.nan
            comparable = False
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "seed": seed,
                "passed": passed,
                "runtime_seconds": max(time.perf_counter() - started, 1e-9),
                "comparable_predictions": comparable,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "error": error,
                "n_sites": n_sites,
                "n_species": n_species,
                "epochs": epochs,
                "full_site_count": full_site_count,
                "full_epochs": full_epochs,
            }
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination, index=False)
    return destination


def evaluate_deepmaxent_gate(
    checkout: Path,
    pilot: Path,
    config: StudyConfig,
) -> GateReport:
    """Evaluate all mandatory inclusion conditions without starting a full run."""

    adapter = DeepMaxentAdapter(checkout)
    formula_passed, formula_reason = adapter.formula_check()
    smoke_passed, smoke_reason = adapter.repository_smoke_check()
    example_passed, example_reason = adapter.repository_tutorial_check()
    if not smoke_passed and example_reason is None:
        example_reason = smoke_reason
        example_passed = False
    pilot_path = Path(pilot)
    projected_days = float("inf")
    pilot_passed = False
    comparable_passed = False
    pilot_reason: str | None = None
    if pilot_path.is_file():
        try:
            rows = pd.read_parquet(pilot_path)
            required = {
                "seed",
                "passed",
                "runtime_seconds",
                "comparable_predictions",
            }
            missing = required.difference(rows.columns)
            if missing:
                raise ValueError(f"pilot is missing columns: {sorted(missing)}")
            runtime = rows.runtime_seconds.to_numpy(dtype=float)
            if (
                rows.empty
                or not np.isfinite(runtime).all()
                or np.any(runtime <= 0)
            ):
                raise ValueError("pilot runtimes must be finite and positive")
            pilot_passed = bool(
                rows.seed.nunique() >= 3
                and rows.passed.astype(bool).all()
            )
            comparable_passed = bool(
                rows.comparable_predictions.astype(bool).all()
            )
            n_full_runs = (
                config.simulation.n_communities
                * len(config.simulation.alignments)
                * len(config.simulation.bias_levels)
                * len(config.background_arms)
            )
            scale = 1.0
            scaling_fields = {
                "n_sites",
                "n_species",
                "epochs",
                "full_site_count",
                "full_epochs",
            }
            if scaling_fields.issubset(rows.columns):
                scale = float(
                    np.median(rows.full_site_count / rows.n_sites)
                    * np.median(config.simulation.n_species / rows.n_species)
                    * np.median(rows.full_epochs / rows.epochs)
                )
            projected_days = float(
                np.median(runtime) * scale * n_full_runs / 86_400
            )
            if not pilot_passed:
                pilot_reason = "pilot did not pass across at least three seeds"
        except (OSError, ValueError, TypeError) as exc:
            pilot_reason = f"pilot artifact is invalid: {exc}"
    else:
        pilot_reason = "pilot artifact is unavailable"

    reasons = [
        reason
        for reason in (formula_reason, example_reason, pilot_reason)
        if reason is not None
    ]
    if not comparable_passed:
        reasons.append("pilot predictions are not comparable truth surfaces")
    if projected_days > 7:
        reasons.append("projected full runtime exceeds seven calendar days")
    commit = adapter.commit
    if commit == "unavailable":
        reasons.append("official repository commit is unavailable")
    include = (
        commit != "unavailable"
        and formula_passed
        and example_passed
        and pilot_passed
        and comparable_passed
        and projected_days <= 7
    )
    return GateReport(
        official_commit=commit,
        formula_check_passed=formula_passed,
        repository_example_passed=example_passed,
        multi_seed_pilot_passed=pilot_passed,
        projected_calendar_days=projected_days,
        comparable_predictions_passed=comparable_passed,
        include=include,
        reasons=tuple(dict.fromkeys(reasons)),
    )
