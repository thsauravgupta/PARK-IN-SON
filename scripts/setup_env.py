# -*- coding: utf-8 -*-
"""
Environment setup and validation for the PPMI pipeline.

WHY this script exists: Before any data download or model training,
we must ensure credentials are present, required packages are importable,
and the project directory tree is in place.  Catching these issues early
(before a multi-hour run) prevents wasted compute and cryptic errors
deep inside the pipeline.

Usage:
    python scripts/setup_env.py
"""

import importlib
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work when running
# this script directly from the command line.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dirs, load_config, seed_everything, setup_logging

# ---------------------------------------------------------------------------
# Reproducibility — always first
# ---------------------------------------------------------------------------
seed_everything(42)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
config: Dict = load_config()
logger = setup_logging(
    name="setup_env",
    level="INFO",
    log_dir=PROJECT_ROOT / config["paths"]["logs"],
)


# ============================================================================
# 1. Validate .env
# ============================================================================

def _validate_dotenv() -> bool:
    """Check that a ``.env`` file exists at the project root and contains
    the required PPMI credential keys.

    WHY: pypmi and ppmi_downloader need username/password to authenticate
    against the LONI IDA portal.  Without them every subsequent script will
    fail.

    Returns:
        True if validation passes; False otherwise.
    """
    env_path: Path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        logger.error(
            ".env file not found at %s.  "
            "Copy .env.example → .env and fill in your PPMI credentials.",
            env_path,
        )
        return False

    # We intentionally do NOT import dotenv at module-level so we can test
    # whether the package is installed at all.
    try:
        from dotenv import dotenv_values  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "python-dotenv is not installed.  "
            "Run:  pip install python-dotenv"
        )
        return False

    values: Dict[str, str | None] = dotenv_values(env_path)
    required_keys: List[str] = ["PPMI_USER", "PPMI_PASSWORD"]
    missing: List[str] = [k for k in required_keys if not values.get(k)]

    if missing:
        logger.error(
            "Missing or empty keys in .env: %s.  "
            "Open .env and set them before proceeding.",
            ", ".join(missing),
        )
        return False

    logger.info(
        ".env validated — PPMI_USER and PPMI_PASSWORD are present."
    )
    return True


# ============================================================================
# 2. Create folder structure
# ============================================================================

def _create_directories() -> None:
    """Create every directory referenced in *config.yaml → paths*.

    WHY: download and processing scripts assume these directories exist.
    Creating them once here avoids scattered ``mkdir`` calls everywhere.
    """
    ensure_dirs(config)
    for key, rel_path in config.get("paths", {}).items():
        logger.info("  ✓ %s → %s", key, PROJECT_ROOT / rel_path)
    logger.info("All project directories created / verified.")


# ============================================================================
# 3. Log system information
# ============================================================================

def _log_system_info() -> None:
    """Log Python version and versions of critical packages.

    WHY: Reproducibility requires knowing the exact software stack.  When
    a colleague reports different metrics, the first question is "what
    versions are you running?".
    """
    logger.info("Python %s on %s", sys.version, platform.platform())

    packages: List[str] = [
        "numpy", "pandas", "scikit-learn", "torch", "yaml",
        "nibabel", "nilearn", "xgboost", "lightgbm",
    ]
    for pkg_name in packages:
        try:
            mod = importlib.import_module(pkg_name)
            version: str = getattr(mod, "__version__", "unknown")
            logger.info("  %-16s %s", pkg_name, version)
        except ImportError:
            logger.warning("  %-16s NOT INSTALLED", pkg_name)


# ============================================================================
# 4. Test PPMI library imports
# ============================================================================

def _test_ppmi_imports() -> Tuple[bool, bool]:
    """Attempt to import pypmi and ppmi_downloader.

    WHY: pypmi (last updated 2020) is fragile and may not install cleanly
    on newer Python.  ppmi_downloader requires Selenium + ChromeDriver.
    Knowing which is available determines the download strategy later.

    Returns:
        Tuple of (pypmi_ok, ppmi_downloader_ok).
    """
    pypmi_ok: bool = False
    downloader_ok: bool = False

    try:
        import pypmi  # type: ignore[import-untyped]  # noqa: F401
        pypmi_ok = True
        logger.info("pypmi imported successfully (v%s).",
                     getattr(pypmi, "__version__", "unknown"))
    except ImportError as exc:
        logger.warning("pypmi is NOT available: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pypmi import raised unexpected error: %s", exc)

    try:
        import ppmi_downloader  # type: ignore[import-untyped]  # noqa: F401
        downloader_ok = True
        logger.info("ppmi_downloader imported successfully (v%s).",
                     getattr(ppmi_downloader, "__version__", "unknown"))
    except ImportError as exc:
        logger.warning("ppmi_downloader is NOT available: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ppmi_downloader import raised unexpected error: %s", exc,
        )

    return pypmi_ok, downloader_ok


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run all environment checks in order and report a summary."""
    logger.info("=" * 72)
    logger.info("PPMI Pipeline — Environment Setup & Validation")
    logger.info("=" * 72)

    # Step 1: .env
    env_ok: bool = _validate_dotenv()

    # Step 2: directories
    _create_directories()

    # Step 3: system info
    _log_system_info()

    # Step 4: PPMI imports
    pypmi_ok, downloader_ok = _test_ppmi_imports()

    # Summary
    logger.info("-" * 72)
    logger.info("SETUP SUMMARY")
    logger.info("  .env valid:            %s", env_ok)
    logger.info("  pypmi available:       %s", pypmi_ok)
    logger.info("  ppmi_downloader avail: %s", downloader_ok)
    logger.info("-" * 72)

    if not env_ok:
        logger.error(
            "Fix .env issues before running download scripts."
        )
        sys.exit(1)

    logger.info("Environment setup complete — ready to download data.")


if __name__ == "__main__":
    main()
