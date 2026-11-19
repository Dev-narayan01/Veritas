"""Artifact discovery, loading and honesty bookkeeping.

An artifact bundle is a directory containing ``manifest.json`` plus the trained
files (``classical.pkl``, ``ppl_gpt2_model.pkl``, ``stacker_<mode>.pkl``,
``thresholds.json``, ``metrics.json`` ...).  Resolution order:

1. ``VERITAS_ARTIFACT_DIR`` when it holds a ``manifest.json``;
2. a Hugging Face Hub snapshot of ``VERITAS_HUB_REPO`` (public repo, no token);
3. nothing — every detector mode then reports itself unavailable.

``artifact_source`` distinguishes a Kaggle/GPU bundle (``kaggle``) from the
CPU-trained baseline (``cpu``).  The manifest recorded at training time decides,
so the UI badge cannot drift from reality.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import MODE_COMPONENTS, MODES, settings
from .detector.classical import ClassicalDetector
from .detector.perplexity import PerplexityDetector
from .detector.stacker import StackerArtifact, load as load_stacker
from .detector.transformer import TransformerDetector
from .logging_setup import get_logger

log = get_logger(__name__)

REQUIRED_FILES = ("manifest.json",)


@dataclass
class ModeState:
    """Availability of one operating mode and why."""

    mode: str
    available: bool
    reason: str | None = None
    components: list[str] = field(default_factory=list)


class ArtifactRegistry:
    """Loads artifacts once and reports which detector modes can run."""

    def __init__(self) -> None:
        self.dir: Path | None = None
        self.manifest: dict[str, Any] = {}
        self.source: str = "none"  # kaggle | cpu | none
        self.notes: list[str] = []
        self.status: str = "idle"  # idle | loading | ready | missing
        self._lock = threading.Lock()

        self._classical: ClassicalDetector | None = None
        self._ppl: PerplexityDetector | None = None
        self._transformer: TransformerDetector | None = None
        self._zero_shot: Any = None
        self._stackers: dict[str, StackerArtifact] = {}
        self._stacker_errors: dict[str, str] = {}
        self._thresholds: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def resolve(self) -> Path | None:
        """Find an artifact directory, downloading from the Hub if configured."""
        local = settings.artifact_path
        if (local / "manifest.json").exists():
            return local
        if settings.hub_repo:
            try:  # pragma: no cover - needs huggingface_hub + network
                from huggingface_hub import snapshot_download

                target = settings.work_dir / "artifacts_hub"
                target.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=settings.hub_repo,
                    local_dir=str(target),
                    token=settings.hf_token or None,
                )
                if (target / "manifest.json").exists():
                    return target
                self.notes.append(f"hub repo {settings.hub_repo} has no manifest.json")
            except Exception as exc:  # noqa: BLE001 - network / auth condition
                self.notes.append(f"hub download failed: {exc}")
                log.warning("hub artifact download failed: %s", exc)
        return None

    def load(self) -> None:
        """Load everything, then compute per-mode availability. Safe to call twice."""
        with self._lock:
            if self.status in {"ready", "missing"}:
                return
            self.status = "loading"
            directory = self.resolve()
            if directory is None:
                self.status = "missing"
                self.source = "none"
                self.notes.append("no artifact bundle found; detector modes unavailable")
                log.warning("no artifacts: AI detection will report unavailable")
                return
            self.dir = directory
            try:
                self.manifest = json.loads((directory / "manifest.json").read_text())
            except json.JSONDecodeError as exc:
                self.notes.append(f"manifest.json unreadable: {exc}")
                self.manifest = {}
            self.source = "kaggle" if self.manifest.get("source") == "kaggle" else "cpu"
            self._load_components(directory)
            self.status = "ready"
            log.info("artifacts loaded from %s (source=%s)", directory, self.source)

    def _load_components(self, directory: Path) -> None:
        classical_path = directory / "classical.pkl"
        self._classical = ClassicalDetector(classical_path)
        self._ppl = PerplexityDetector(directory / "ppl_gpt2_model.pkl")
        # The window travels with the bundle: serving a checkpoint trained at 1024
        # through a 256-token default would truncate exactly the long documents the
        # model was trained to handle. Falls back to the env setting when absent.
        served_max_len = self.manifest.get("transformer_max_len") or settings.transformer_max_len
        self._transformer = TransformerDetector(
            directory / "transformer", max_len=int(served_max_len)
        )

        for mode in MODES:
            path = directory / f"stacker_{mode}.pkl"
            if not path.exists():
                self._stacker_errors[mode] = f"stacker missing: {path.name}"
                continue
            try:
                self._stackers[mode] = load_stacker(path)
            except Exception as exc:  # noqa: BLE001 - corrupt artifact
                self._stacker_errors[mode] = f"stacker failed to load: {exc}"

        thresholds_path = directory / "thresholds.json"
        if thresholds_path.exists():
            try:
                self._thresholds = json.loads(thresholds_path.read_text())
            except json.JSONDecodeError as exc:
                self.notes.append(f"thresholds.json unreadable: {exc}")

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    def component_available(self, name: str) -> tuple[bool, str | None]:
        """Return (available, reason) for a single component."""
        self.load()
        if self.dir is None:
            return False, "no artifact bundle is loaded on this server"
        if name == "classical":
            assert self._classical is not None
            return self._classical.available, self._classical.reason
        if name == "ppl_gpt2":
            assert self._ppl is not None
            return self._ppl.available, self._ppl.reason
        if name == "transformer":
            assert self._transformer is not None
            return self._transformer.available, self._transformer.reason
        if name in {"binoculars", "fastdetect"}:
            from .detector.optional import zero_shot_available

            ok, reason = zero_shot_available(name)
            return ok, reason
        return False, f"unknown component {name}"

    def mode_state(self, mode: str) -> ModeState:
        """A mode is available only when every component and its stacker load.

        Components are checked before the stacker on purpose. Both are missing for the
        heavy modes in a CPU bundle, and the component is the blocker a reader can act on
        ("install torch"), whereas "stacker missing" would send them looking in the wrong
        place. The first *fatal* reason wins: a stacker is useless without its inputs.
        """
        self.load()
        components = list(MODE_COMPONENTS[mode])
        for component in components:
            ok, reason = self.component_available(component)
            if not ok:
                return ModeState(mode, False, f"{component}: {reason}", components)
        if mode in self._stacker_errors:
            return ModeState(mode, False, self._stacker_errors[mode], components)
        if mode not in self._stackers:
            return ModeState(mode, False, f"stacker_{mode}.pkl not found", components)
        return ModeState(mode, True, None, components)

    def all_modes(self) -> list[ModeState]:
        return [self.mode_state(mode) for mode in MODES]

    def default_mode(self) -> str | None:
        """The configured default if available, else the cheapest available mode."""
        state = self.mode_state(settings.default_mode)
        if state.available:
            return settings.default_mode
        for mode in ("standard", "fast", "lite"):
            if self.mode_state(mode).available:
                return mode
        return None

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    @property
    def classical(self) -> ClassicalDetector:
        self.load()
        assert self._classical is not None
        return self._classical

    @property
    def ppl(self) -> PerplexityDetector:
        self.load()
        assert self._ppl is not None
        return self._ppl

    @property
    def transformer(self) -> TransformerDetector:
        self.load()
        assert self._transformer is not None
        return self._transformer

    def stacker(self, mode: str) -> StackerArtifact:
        self.load()
        return self._stackers[mode]

    @property
    def thresholds(self) -> dict[str, Any]:
        self.load()
        return self._thresholds

    @property
    def data_source(self) -> str:
        self.load()
        return str(self.manifest.get("data_source", "unknown"))

    def artifact_file(self, name: str) -> Path | None:
        """Path to a JSON/plot artifact (used by ``/api/metrics``)."""
        self.load()
        if self.dir is None:
            return None
        candidate = self.dir / name
        return candidate if candidate.exists() else None

    def adopt(self, source_dir: Path) -> None:
        """Copy a verified bundle into the live artifact directory."""
        target = settings.artifact_path
        target.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            dest = target / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        self.status = "idle"
        self.load()


registry = ArtifactRegistry()
