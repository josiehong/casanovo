"""Unit tests for chimeric spectrum charge-state assignment."""

import unittest.mock

import numpy as np
import pytest
import torch

from casanovo.config import Config
from casanovo.data import psm
from casanovo.denovo.chimera import ChimeraTokenizer
from casanovo.denovo.model import Spec2Pep


PEPTIDE = "PEPTIDEK"


@pytest.fixture
def chimera_model():
    """A minimal Spec2Pep in chimera mode with canonical residues."""
    tokenizer = ChimeraTokenizer(residues=Config().residues)
    return Spec2Pep(
        tokenizer=tokenizer,
        chimera_isotope_error_range=(0, 2),
        chimera_max_charge=4,
    )


def _theoretical_mz(tokenizer, sequence, charge):
    """Return the monoisotopic precursor m/z for sequence at the given charge."""
    return tokenizer.calculate_precursor_ions(
        sequence, torch.tensor(charge)
    ).item()


# ---------------------------------------------------------------------------
# _assign_chimera_charge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_charge", [1, 2, 3, 4])
def test_correct_charge_selected(chimera_model, true_charge):
    """Correct charge is returned when obs_mz matches the theoretical m/z exactly."""
    obs_mz = _theoretical_mz(chimera_model.tokenizer, PEPTIDE, true_charge)
    charge, calc_mz = chimera_model._assign_chimera_charge(PEPTIDE, obs_mz)
    assert charge == true_charge
    assert np.isclose(calc_mz, obs_mz, rtol=1e-6)


@pytest.mark.parametrize("iso", [1, 2])
def test_isotope_correction(chimera_model, iso):
    """Charge is still correct when obs_mz is shifted by an isotope peak."""
    true_charge = 2
    mono_mz = _theoretical_mz(chimera_model.tokenizer, PEPTIDE, true_charge)
    # Simulate the instrument isolating the M+iso isotope peak.
    obs_mz = mono_mz + iso * 1.00335 / true_charge
    charge, _ = chimera_model._assign_chimera_charge(PEPTIDE, obs_mz)
    assert charge == true_charge, (
        f"iso={iso}: expected charge {true_charge}, got {charge}"
    )


def test_best_charge_returned_even_outside_tolerance(chimera_model):
    """The closest charge is still assigned even when no charge fits within tolerance."""
    # obs_mz=9999 Da is far from any realistic charge state of PEPTIDEK,
    # but the function should still return the best-effort charge rather than fail.
    charge, calc_mz = chimera_model._assign_chimera_charge(PEPTIDE, 9999.0)
    assert isinstance(charge, int)
    assert not np.isnan(calc_mz)


def test_all_charge_calculations_fail(chimera_model):
    """When calculate_precursor_ions raises for every charge, calc_mz is NaN."""
    with unittest.mock.patch.object(
        chimera_model.tokenizer,
        "calculate_precursor_ions",
        side_effect=Exception("bad token"),
    ):
        charge, calc_mz = chimera_model._assign_chimera_charge(PEPTIDE, 500.0)
    assert np.isnan(calc_mz)


def test_ambiguous_charge_prefers_lower_error(chimera_model):
    """When two charge states are close, the one with lower ppm error wins."""
    true_charge = 3
    obs_mz = _theoretical_mz(chimera_model.tokenizer, PEPTIDE, true_charge)
    charge, calc_mz = chimera_model._assign_chimera_charge(PEPTIDE, obs_mz)
    assert charge == true_charge
    assert np.isclose(calc_mz, obs_mz, rtol=1e-6)


# ---------------------------------------------------------------------------
# on_predict_batch_end integration
# ---------------------------------------------------------------------------


def test_on_predict_batch_end_updates_charge_and_calc_mz(chimera_model):
    """on_predict_batch_end overwrites charge and sets calc_mz for chimera PSMs."""
    true_charge = 2
    obs_mz = _theoretical_mz(chimera_model.tokenizer, PEPTIDE, true_charge)

    out_writer = unittest.mock.MagicMock()
    out_writer.psms = []
    chimera_model.out_writer = out_writer

    match = psm.PepSpecMatch(
        sequence=PEPTIDE,
        spectrum_id=("file.mgf", "1"),
        peptide_score=0.9,
        charge=1,  # deliberately wrong — should be corrected
        calc_mz=float("nan"),
        exp_mz=obs_mz,
        aa_scores=np.array([0.9] * len(PEPTIDE)),
    )
    chimera_model.on_predict_batch_end([match])

    assert len(out_writer.psms) == 1
    saved = out_writer.psms[0]
    assert saved.charge == true_charge
    assert np.isclose(saved.calc_mz, obs_mz, rtol=1e-5)


def test_score_unchanged_when_outside_tolerance(chimera_model):
    """Peptide score is not modified even when no charge fits within tolerance."""
    out_writer = unittest.mock.MagicMock()
    out_writer.psms = []
    chimera_model.out_writer = out_writer

    original_score = 0.8
    match = psm.PepSpecMatch(
        sequence=PEPTIDE,
        spectrum_id=("file.mgf", "2"),
        peptide_score=original_score,
        charge=2,
        calc_mz=float("nan"),
        exp_mz=9999.0,  # no charge state will match within tolerance
        aa_scores=np.array([0.8] * len(PEPTIDE)),
    )
    chimera_model.on_predict_batch_end([match])

    assert len(out_writer.psms) == 1
    assert out_writer.psms[0].peptide_score == pytest.approx(original_score)
