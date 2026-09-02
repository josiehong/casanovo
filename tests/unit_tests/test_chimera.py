"""Unit tests for chimeric two-peptide sequencing."""

import collections
import pickle

import numpy as np
import pytest
import torch

from depthcharge.tokenizers import PeptideTokenizer

from casanovo.config import Config
from casanovo.data import ms_io, psm
from casanovo.denovo.chimera import ChimeraTokenizer, MskbChimeraTokenizer
from casanovo.denovo.model import H2O_MASS, PROTON_MASS, Spec2Pep
from casanovo.denovo.model_runner import ModelRunner


def _tokenizer():
    """A chimera tokenizer over the default residue alphabet."""
    return ChimeraTokenizer(
        residues=Config().residues,
        reverse=True,
        start_token=None,
        stop_token="$",
    )


def _model(**kwargs):
    """A small chimeric model."""
    params = dict(
        dim_model=8,
        n_head=2,
        dim_feedforward=8,
        n_layers=1,
        max_peptide_len=20,
        tokenizer=_tokenizer(),
        chimera=True,
    )
    params.update(kwargs)
    return Spec2Pep(**params)


def _tokens(tokenizer, peptide):
    """The token indices of a peptide, without padding."""
    return [
        int(t) for t in tokenizer.tokenize([peptide], add_stop=False)[0] if t
    ]


def _neutral_mass(tokenizer, peptide):
    """The neutral mass of a peptide."""
    return (
        sum(tokenizer.residues[aa] for aa in tokenizer.split(peptide))
        + H2O_MASS
    )


def test_split_annotation():
    """A chimeric annotation splits into its two peptides."""
    tokenizer = _tokenizer()
    assert tokenizer.split_annotation("PEPTIDEK:LLGGSSAAR") == (
        "PEPTIDEK",
        "LLGGSSAAR",
    )
    # A single peptide leaves the second slot empty.
    assert tokenizer.split_annotation("PEPTIDEK") == ("PEPTIDEK", "")


def test_split_annotation_ignores_modifications():
    """A separator inside a modification is not a separator.

    The same character names a controlled-vocabulary modification inside
    brackets, so the split has to ignore those.
    """
    tokenizer = _tokenizer()
    assert tokenizer.split_annotation(
        "[+25.980265]-PEPTIDEK:M[Oxidation]LLR"
    ) == ("[+25.980265]-PEPTIDEK", "M[Oxidation]LLR")
    assert tokenizer.split_annotation("M[UNIMOD:35]PEPTIDEK:LLGGSSAAR") == (
        "M[UNIMOD:35]PEPTIDEK",
        "LLGGSSAAR",
    )
    assert tokenizer.split_annotation("M[UNIMOD:35]PEPTIDEK") == (
        "M[UNIMOD:35]PEPTIDEK",
        "",
    )


def test_split_annotation_rejects_three_peptides():
    """Only two peptides fit in the two slots."""
    with pytest.raises(ValueError, match="at most one"):
        _tokenizer().split_annotation("PEPTIDEK:LLGGSSAAR:AAAAAAK")


def test_chimera_adds_no_residue():
    """The separator is not a residue, so the alphabet is unchanged.

    This is what lets the mass search run over a chimeric model untouched,
    and what keeps a plain checkpoint loadable as a warm start.
    """
    residues = Config().residues
    kwargs = dict(reverse=True, start_token=None, stop_token="$")
    chimeric = ChimeraTokenizer(residues=residues, **kwargs)
    plain = PeptideTokenizer(residues=residues, **kwargs)

    assert ":" not in chimeric.index
    assert chimeric.index == plain.index
    assert chimeric.residues == plain.residues


@pytest.mark.parametrize(
    "tokenizer_class", [ChimeraTokenizer, MskbChimeraTokenizer]
)
def test_tokenizer_is_picklable(tokenizer_class):
    """The tokenizer survives being written into a checkpoint.

    Lightning pickles it into the hyperparameters and drops what it cannot
    pickle without failing, which would silently remove the tokenizer from
    the checkpoint.
    """
    tokenizer = tokenizer_class(
        residues=Config().residues,
        reverse=True,
        start_token=None,
        stop_token="$",
    )
    assert pickle.loads(pickle.dumps(tokenizer)).index == tokenizer.index


def test_terminal_modification_repair():
    """An absent terminal modification produces no token.

    Pyteomics returns an empty list rather than None, which depthcharge
    turns into a spurious "[+0.000000]-" token outside the alphabet.
    """
    assert "[+0.000000]-" not in _tokenizer().split("PEPTIDEK")


def test_frames_scale_with_chimera():
    """Each slot gets a full max_peptide_len frames."""
    single = _model(chimera=False)
    assert single.n_decoder_frames == single.max_peptide_len

    chimeric = _model(chimera=True)
    assert chimeric.n_decoder_frames == 2 * chimeric.max_peptide_len
    # The decoder prepends a global precursor token, so slot A is one
    # frame longer than slot B.
    frames = chimeric.n_decoder_frames + 1
    assert chimeric.chimera_split == chimeric.max_peptide_len + 1
    assert frames - chimeric.chimera_split == chimeric.max_peptide_len


def test_single_loss_matches_builtin_reduction():
    """Non-chimeric training is unchanged by the chimera code."""
    model = _model(chimera=False)
    tokenizer = model.tokenizer
    truth = tokenizer.tokenize(
        ["PEPTIDEK", "LLGGSSAAR", "AAAAAAK"], add_stop=True
    )
    torch.manual_seed(0)
    pred = torch.randn(
        truth.shape[0], model.n_decoder_frames + 1, model.vocab_size
    )

    ours, aux = model._single_loss(pred, [], truth)
    assert aux is None

    target_lengths = (truth != 0).sum(dim=1)
    builtin = model.ctc_loss(
        pred.log_softmax(-1).transpose(0, 1),
        truth,
        torch.full((truth.shape[0],), pred.shape[1], dtype=torch.long),
        target_lengths,
    )
    assert torch.allclose(ours, builtin)


def test_chimera_loss_takes_the_better_assignment():
    """Nothing ties a peptide to a slot, so both orders are scored."""
    model = _model()
    tokenizer = model.tokenizer
    truth_a = tokenizer.tokenize(["PEPTIDEK", "LLGGSSAAR"], add_stop=True)
    truth_b = tokenizer.tokenize(["LLGGSSAAR", "PEPTIDEK"], add_stop=True)
    torch.manual_seed(0)
    pred = torch.randn(2, model.n_decoder_frames + 1, model.vocab_size)

    loss, _ = model._chimera_loss(pred, [], {"seq": truth_a, "seq_2": truth_b})

    slot_a = pred[:, : model.chimera_split]
    slot_b = pred[:, model.chimera_split :]
    len_a = (truth_a != 0).sum(dim=1)
    len_b = (truth_b != 0).sum(dim=1)
    direct = model._ctc_per_spectrum(
        slot_a, truth_a, len_a
    ) + model._ctc_per_spectrum(slot_b, truth_b, len_b)
    swapped = model._ctc_per_spectrum(
        slot_a, truth_b, len_b
    ) + model._ctc_per_spectrum(slot_b, truth_a, len_a)
    denominator = (len_a + len_b).clamp(min=1)

    assert torch.allclose(
        loss, (torch.minimum(direct, swapped) / denominator).mean()
    )
    assert loss <= (direct / denominator).mean()
    assert loss <= (swapped / denominator).mean()


def test_chimera_loss_with_an_empty_slot():
    """A spectrum with one peptide trains the other slot to stay empty."""
    model = _model()
    truth_a = model.tokenizer.tokenize(["PEPTIDEK"], add_stop=True)
    empty = torch.zeros((1, 1), dtype=torch.long)
    torch.manual_seed(0)
    pred = torch.randn(1, model.n_decoder_frames + 1, model.vocab_size)

    loss, _ = model._chimera_loss(pred, [], {"seq": truth_a, "seq_2": empty})
    assert torch.isfinite(loss)


def test_intermediates_reuse_the_final_assignment():
    """Every self-conditioning layer scores the same slot assignment.

    Letting each layer choose its own would feed a prediction forward
    under one assignment and read it back under another.
    """
    model = _model()
    tokenizer = model.tokenizer
    batch = {
        "seq": tokenizer.tokenize(["PEPTIDEK"], add_stop=True),
        "seq_2": tokenizer.tokenize(["LLGGSSAAR"], add_stop=True),
    }
    torch.manual_seed(0)
    pred = torch.randn(1, model.n_decoder_frames + 1, model.vocab_size)
    intermediates = [
        torch.randn(1, model.n_decoder_frames + 1, model.vocab_size)
        for _ in range(2)
    ]

    alone, no_aux = model._chimera_loss(pred, [], batch)
    combined, aux = model._chimera_loss(pred, intermediates, batch)

    assert no_aux is None
    assert aux is not None
    # The final layer's loss is what gets logged, so it must not shift
    # when self-conditioning is switched on.
    assert torch.allclose(alone, combined)


def test_best_effort_charge_needs_a_charge_range():
    """Without a charge range the annotated charge is the only candidate."""
    model = _model(charge_range=None)
    tokens = _tokens(model.tokenizer, "PEPTIDEK")
    mass = _neutral_mass(model.tokenizer, "PEPTIDEK")
    assert (
        model._best_effort_charge(tokens, mass, mass / 2 + PROTON_MASS, 2) == 2
    )


def test_best_effort_charge_finds_another_charge():
    """A sub-peptide gets the charge it comes nearest to matching.

    The annotated charge belongs to whichever precursor the instrument
    picked; the other co-isolated peptide generally has a different one.
    """
    model = _model(charge_range=(1, 4))
    tokenizer = model.tokenizer
    # An observed m/z that the annotated peptide explains at charge 2.
    annotated_mass = _neutral_mass(tokenizer, "PEPTIDEK")
    observed_mz = annotated_mass / 2 + PROTON_MASS
    # A heavier peptide at the same m/z has to be more highly charged.
    heavy = "LLGGSSAARLLGGSSAARLLGG"
    charge = model._best_effort_charge(
        _tokens(tokenizer, heavy), annotated_mass, observed_mz, 2
    )
    assert charge > 2


def test_matching_window_accepts_slots_at_different_charges():
    """One spectrum, two peptides, two charges.

    This is what a chimeric spectrum needs: the annotated charge belongs
    to whichever precursor the instrument picked, and the co-isolated
    peptide generally carries a different one at the same observed m/z.
    """
    tokenizer = _tokenizer()
    light = "PEPTIDEK"
    annotated_mass = _neutral_mass(tokenizer, light)
    observed_mz = annotated_mass / 2 + PROTON_MASS

    # The residue mass that same m/z implies at charge 3, approached with
    # glycines so the leftover is known exactly.
    glycine = tokenizer.residues["G"]
    target = (observed_mz - PROTON_MASS) * 3 - H2O_MASS
    n_glycine = round(target / glycine)
    heavy = [tokenizer.index["G"]] * n_glycine
    leftover = abs(target - n_glycine * glycine)
    # Wide enough to absorb that leftover, and far narrower than the
    # several hundred Da between one candidate charge and the next.
    tolerance = 1.5 * leftover / target * 1e6

    model = _model(
        charge_range=(1, 4),
        precursor_mass_tol=tolerance,
        isotope_error_range=(0, 0),
    )
    light_window = model._matching_window(
        _tokens(tokenizer, light), annotated_mass, observed_mz, 2
    )
    heavy_window = model._matching_window(
        heavy, annotated_mass, observed_mz, 2
    )

    assert light_window is not None and light_window[0] == 2
    assert heavy_window is not None and heavy_window[0] == 3


def _match(sequence, score):
    """A PSM with the given sequence and score."""
    return psm.PepSpecMatch(
        sequence=sequence,
        spectrum_id=("file", "1"),
        peptide_score=score,
        charge=2,
        calc_mz=np.nan,
        exp_mz=500.0,
        aa_scores=np.array([score]),
    )


def test_select_chimera_keeps_both_slots():
    """Two different peptides are both reported for the spectrum."""
    model = _model()
    kept = model._select_chimera(
        [_match("PEPTIDEK", 0.9), _match("LLGGSSAAR", 0.8)]
    )
    assert [m.sequence for m in kept] == ["PEPTIDEK", "LLGGSSAAR"]
    assert all(m.spectrum_id == ("file", "1") for m in kept)


def test_select_chimera_deduplicates():
    """Both slots can converge; keep the higher-scoring copy."""
    model = _model()
    kept = model._select_chimera(
        [_match("PEPTIDEK", 0.4), _match("PEPTIDEK", 0.9)]
    )
    assert len(kept) == 1
    assert kept[0].peptide_score == 0.9


def test_select_chimera_drops_short_peptides():
    """A slot below min_peptide_len is not reported."""
    model = _model(min_peptide_len=6)
    kept = model._select_chimera([_match("PEPTIDEK", 0.9), _match("AAK", 0.8)])
    assert [m.sequence for m in kept] == ["PEPTIDEK"]


def test_select_chimera_drops_internal_nterm_mods():
    """An N-terminal modification past the start is not valid ProForma."""
    model = _model()
    kept = model._select_chimera(
        [_match("PEPTIDEK", 0.9), _match("PEP[Acetyl]-TIDEK", 0.8)]
    )
    assert [m.sequence for m in kept] == ["PEPTIDEK"]


def test_build_psm_returns_none_when_nothing_decoded():
    """An empty slot produces no PSM."""
    model = _model()
    assert model._build_psm([], [], 2, ("file", "1"), 500.0) is None


def test_chimeric_data_and_prediction(
    tmp_path, mgf_chimera, tiny_config_chimera
):
    """The whole chimeric path, from annotation file to PSMs.

    Covers what the unit tests above cannot: that the dataset splits the
    annotation into two targets, that a single-peptide spectrum leaves the
    second empty, and that prediction reports one PSM per slot under one
    spectrum identifier.
    """
    config = Config(tiny_config_chimera)
    config.max_epochs = 1
    config.n_layers = 1
    config.dim_model = 32
    config.dim_feedforward = 32
    config.lance_dir = str(tmp_path / "lance")

    runner = ModelRunner(config=config, output_dir=tmp_path)
    runner.initialize_trainer(train=True)
    runner.initialize_tokenizer()
    runner.initialize_model(train=True)
    runner.initialize_data_module([str(mgf_chimera)], [str(mgf_chimera)])
    runner.loaders.setup()
    batch = next(iter(runner.loaders.train_dataloader()))

    # Both targets reach the model.
    assert "seq_2" in batch
    truths_b = runner.model._detokenize_targets(batch["seq_2"])
    # The fixture's last spectrum names one peptide, so its second slot is
    # empty; the other two name two.
    assert truths_b[-1] == ""
    assert all(truths_b[i] for i in range(len(truths_b) - 1))

    assert torch.isfinite(runner.model.training_step(batch))

    psms = runner.model.predict_step(batch)
    n_spectra = batch["seq"].shape[0]
    # At most one PSM per slot, and the pair shares a spectrum id.
    assert 0 < len(psms) <= 2 * n_spectra
    by_spectrum = collections.Counter(p.spectrum_id for p in psms)
    assert all(count <= 2 for count in by_spectrum.values())
    for spectrum_id, count in by_spectrum.items():
        sequences = {p.sequence for p in psms if p.spectrum_id == spectrum_id}
        assert len(sequences) == count  # deduplicated

    # The reported charge has to yield a calculated m/z downstream.
    runner.model.out_writer = ms_io.MztabWriter(str(tmp_path / "out.mztab"))
    runner.model.on_predict_batch_end(psms)
    assert len(runner.model.out_writer.psms) == len(psms)
    assert all(np.isfinite(p.calc_mz) for p in runner.model.out_writer.psms)
