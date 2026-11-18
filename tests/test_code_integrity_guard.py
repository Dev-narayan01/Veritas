"""The Windows code-integrity guard in ml/train_modernbert.py.

Why this has tests: a machine with Smart App Control enabled cannot load PyTorch's
unsigned CUDA DLLs, and the failure arrives as an OSError inside an import, which
reads like a broken install. The script used to answer that with "reinstall from the
CUDA index", which cannot work: the policy refuses the file wherever it came from.
Now it names the policy instead.

The negative cases carry most of the weight. Calling an ordinary failure a policy
block would send someone with a genuinely damaged install off to disable a security
feature, so `Error loading ... or one of its dependencies` and `Bad Image` are
deliberately not matches.
"""

from __future__ import annotations

import pytest

from ml.train_modernbert import _CODE_INTEGRITY_HELP, _chain_has_code_integrity_block


def _oserror_4551(message: str) -> OSError:
    """OSError the way Windows raises it, with .winerror populated."""
    exc = OSError(4551, message)
    exc.winerror = 4551
    return exc


def test_user_reported_error_is_recognised() -> None:
    """The literal message from a real RTX 3050 machine, minus the machine path."""
    exc = _oserror_4551(
        "An Application Control policy has blocked this file. "
        'Error loading "torch\\lib\\c10_cuda.dll" or one of its dependencies.'
    )
    assert _chain_has_code_integrity_block(exc)


def test_block_is_found_through_a_wrapping_exception() -> None:
    """Python wraps blocked .pyd loads in ImportError, so the cause carries the code."""
    inner = _oserror_4551("An Application Control policy has blocked this file.")
    outer = ImportError("DLL load failed while importing _C")
    outer.__cause__ = inner
    assert _chain_has_code_integrity_block(outer)


def test_winerror_survives_message_replacement() -> None:
    assert _chain_has_code_integrity_block(_oserror_4551("whatever windows said"))


@pytest.mark.parametrize(
    "message",
    [
        "An Application Control policy has blocked this file.",
        "did not meet the Enterprise signing level requirements",
        "status 0xc0e90002",
    ],
)
def test_message_wording_alone_is_enough(message: str) -> None:
    """winerror is absent from Event Viewer text and from some re-raised errors."""
    assert _chain_has_code_integrity_block(Exception(message))


@pytest.mark.parametrize(
    "message",
    [
        'Error loading "torch\\lib\\torch.dll" or one of its dependencies.',
        "Bad Image",
        "The specified module could not be found",
        "",
    ],
)
def test_ordinary_failures_are_not_mistaken_for_policy_blocks(message: str) -> None:
    """A corrupt wheel prints the same "Error loading" line; there a reinstall IS right."""
    assert not _chain_has_code_integrity_block(Exception(message))


def test_missing_module_is_not_a_policy_block() -> None:
    assert not _chain_has_code_integrity_block(ImportError("No module named 'torch'"))


def test_winerror_126_is_not_a_policy_block() -> None:
    exc = OSError(126, "The specified module could not be found")
    exc.winerror = 126
    assert not _chain_has_code_integrity_block(exc)


def test_self_referential_context_terminates() -> None:
    """Exception chains can be circular; the walk must not hang."""
    exc = Exception("x")
    exc.__context__ = exc
    assert not _chain_has_code_integrity_block(exc)


def test_help_text_does_not_promise_a_reinstall() -> None:
    """The whole point: never tell the user that pulling this lever again will help."""
    lowered = _CODE_INTEGRITY_HELP.lower()
    assert "will not clear it" in lowered
    assert "wsl2" in lowered
    assert "cpu" in lowered
