"""A de novo peptide sequencing model."""

import collections
import heapq
import itertools
import logging
import warnings
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import einops
import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.tokenizers import PeptideTokenizer

from .. import config
from ..data import ms_io, psm
from ..denovo.transformers import PeptideDecoder, SpectrumEncoder
from . import evaluate
from .muon import MuonWithAuxAdamW

logger = logging.getLogger("casanovo")

H2O_MASS = 18.010565
ISOTOPE_SPACING = 1.00335
PROTON_MASS = 1.007276
# Precise mass control (PMC) decoding settings: the mass discretization
# step, the headroom added to the mass axis so that rounding cannot carry
# a valid path off the end of it, a cap on the backtracking table size,
# and the minimum frame probability for a token to take part in a frame
# at all (see `_pmc_decode`: it buys roughly 11x, since CTC output is
# blank-dominated and only a handful of frames per spectrum carry a
# plausible residue).
PMC_RESOLUTION = 0.01
PMC_MASS_GUARD = 0.1
PMC_MAX_POINTER_BYTES = 1_000_000_000
PMC_MIN_EMIT_PROB = 1e-4


class Spec2Pep(pl.LightningModule):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must
        be divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the
        transformer model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The
        remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value. If ``None``, the intensity will be
        projected up to ``dim_model`` using a linear layer, then summed
        with the m/z encoding for each peak.
    max_peptide_len : int
        The maximum peptide length to decode.
    residues : str | Dict[str, float]
        The amino acid dictionary and their masses. By default
        ("canonical") this is only the 20 canonical amino acids, with
        cysteine carbamidomethylated. If "massivekb", this dictionary
        will include the modifications found in MassIVE-KB.
        Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float
        The maximum allowable precursor mass tolerance (in ppm) for
        correct predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a
        non-monoisotopic peak for fragmentation by not penalizing
        predicted precursor m/z's that fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    charge_range : Optional[Tuple[int, int]]
        Precursor charges to consider during precise mass control
        decoding, as an inclusive `(min, max)` pair. `None` (the
        default) trusts the charge annotated on the spectrum and is
        exactly the previous behaviour. A range instead treats the
        annotation as possibly wrong: the neutral mass is recomputed
        from the observed m/z at every charge in the range, a peptide is
        accepted if it matches under any of them, and the charge that
        matched is the one reported for the PSM. Widening the range
        widens the mass axis the DP has to span, which can force a
        coarser discretization for heavy precursors (see `_pmc_decode`).
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams : int
        Number of beams used during beam search decoding.
    top_match : int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    train_label_smoothing : float
        Unused; retained for configuration compatibility. The CTC loss
        does not support label smoothing.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    out_writer : ms_io.MztabWriter | None
        The output writer for the prediction results.
    calculate_precision : bool
        Calculate the validation set precision during training.
        This is expensive.
    tokenizer: PeptideTokenizer | None
        Tokenizer object to process peptide sequences.
    chimera : bool
        Sequence two co-fragmented peptides per spectrum, one in each half
        of the decoder frames, instead of a single peptide across all of
        them.
    **kwargs : Dict
        Additional keyword arguments for the optimizer: ``muon_lr`` and
        ``muon_momentum`` configure the Muon parameter group; the rest
        are passed to the auxiliary AdamW group.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        max_peptide_len: int = 100,
        residues: str | Dict[str, float] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        charge_range: Optional[Tuple[int, int]] = None,
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        tokenizer: PeptideTokenizer | None = None,
        chimera: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer or PeptideTokenizer()
        # Vocabulary: tokenizer tokens incl. padding (0), plus a
        # dedicated CTC blank class as the last index.
        self.vocab_size = len(self.tokenizer) + 2
        self.blank_token = self.vocab_size - 1
        # Build the model.
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )
        # Self-conditioning: which decoder layers score their own hidden
        # states and feed the prediction forward, and how much of the loss
        # those auxiliary predictions carry. Empty list disables it, and the
        # model is then identical to the plain CTC decoder.
        self.self_cond_layers = tuple(kwargs.pop("self_cond_layers", ()) or ())
        self.self_cond_weight = float(kwargs.pop("self_cond_weight", 0.5))
        self.decoder = PeptideDecoder(
            n_tokens=self.tokenizer,
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            max_charge=max_charge,
            self_cond_layers=self.self_cond_layers,
        )
        self.softmax = torch.nn.Softmax(2)
        self.ctc_loss = torch.nn.CTCLoss(
            blank=self.blank_token, zero_infinity=True
        )
        # Chimeric training needs the loss of each spectrum on its own, to
        # pick the better of the two ways to assign the peptides to the
        # slots before reducing over the batch.
        self.ctc_loss_per_seq = torch.nn.CTCLoss(
            blank=self.blank_token, zero_infinity=True, reduction="none"
        )
        self._ctc_infeasible_warned = False
        self._pmc_size_warned = False
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.muon_lr = kwargs.pop("muon_lr", 0.02)
        self.muon_momentum = kwargs.pop("muon_momentum", 0.95)
        # `kwargs` will contain additional arguments as well as
        # unrecognized arguments, including deprecated ones. Remove the
        # deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_peptide_len = max_peptide_len
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.charge_range = (
            None if charge_range is None else tuple(charge_range)
        )
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.stop_token = self.tokenizer.stop_int

        # Chimeric sequencing: the decoder frames are split into two slots,
        # one peptide each, instead of carrying a single peptide. Each slot
        # gets a full `max_peptide_len` frames, so that setting keeps meaning
        # the longest peptide the model can emit and the decoder simply runs
        # twice as wide. The decoder prepends a global precursor token, so
        # slot A is everything before `chimera_split`, one frame longer than
        # slot B, which is everything from it on.
        #
        # Splitting the frames rather than marking the boundary with a
        # separator residue keeps the alphabet, and so the mass axis PMC
        # searches, identical to single-peptide sequencing, and lets PMC run
        # per slot with no changes.
        self.chimera = chimera
        self.n_decoder_frames = self.max_peptide_len * (2 if chimera else 1)
        self.chimera_split = 1 + self.max_peptide_len

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []

        # Output writer during predicting.
        self.out_writer = out_writer

        # Get n-term mod tokens
        self.n_term = [
            aa
            for aa in self.tokenizer.index
            if aa.startswith("[") and aa.endswith("]-")
        ]
        # Register tensor buffers for negative mass amino acid indices
        self.register_buffer(
            "neg_mass_idx",
            torch.tensor(
                [
                    self.tokenizer.index[aa]  # all negative‑mass AAs
                    for aa, mass in self.tokenizer.residues.items()
                    if mass < 0
                ],
                dtype=torch.int,
            ),
            persistent=False,
        )

        # Register tensor buffer for N-terminal modification indices
        self.register_buffer(
            "nterm_idx",
            torch.tensor(
                [self.tokenizer.index[aa] for aa in self.n_term],
                dtype=torch.int,
            ),
            persistent=False,
        )

        # Register tensor buffer for amino acid token masses
        self.register_buffer(
            "token_masses",
            torch.zeros(self.vocab_size, dtype=torch.float64),
            persistent=False,
        )
        # Populate token masses from tokenizer residues
        for aa, mass in self.tokenizer.residues.items():
            idx = self.tokenizer.index.get(aa)
            if idx is not None:
                self.token_masses[idx] = mass

    @property
    def device(self) -> torch.device:
        """
        The device on which the model is currently running.

        Returns
        -------
        torch.device
            The device on which the model is currently running.
        """
        return next(self.parameters()).device

    def _process_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a SpectrumDataset batch to tensors.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        mzs : torch.Tensor of shape (batch_size, max_peaks)
            The m/z values for each spectrum.
        intensities : torch.Tensor of shape (batch_size, max_peaks)
            The intensity values for each spectrum.
        precursors : torch.Tensor of shape (batch_size, 3)
            A tensor with the precursor neutral mass, precursor charge,
            and precursor m/z.
        seqs : np.ndarray
            The spectrum identifiers (during de novo sequencing) or
            peptide sequences (during training).
        """
        precursor_mzs = batch["precursor_mz"].squeeze(0)
        precursor_charges = batch["precursor_charge"].squeeze(0)
        precursor_masses = (precursor_mzs - PROTON_MASS) * precursor_charges
        precursors = torch.vstack(
            [precursor_masses, precursor_charges, precursor_mzs]
        ).T

        mzs = batch["mz_array"]
        intensities = batch["intensity_array"]
        seqs = batch.get("seq")

        return mzs, intensities, precursors, seqs

    def forward(self, batch):
        return self._forward_step(batch)

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step for non-autoregressive decoding.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset.

        Returns
        -------
        scores : torch.Tensor of shape
                (n_spectra, max_peptide_len + 1, n_amino_acids)
            The frame-level amino acid scores for each prediction.
        seqs : torch.Tensor of shape (n_spectra, length) or None
            The ground truth tokens for training, or None for inference.
        intermediates : list of torch.Tensor
            Only when ``return_intermediates``: the scores from each
            self-conditioning layer, for the auxiliary CTC losses. Empty
            unless ``self_cond_layers`` is set.
        """
        mzs, ints, precursors, seqs = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)

        # Decode a fixed number of frames; the CTC loss aligns them to
        # the (shorter) ground truth peptide. Chimeric sequencing decodes
        # two peptides, so it needs twice the frames.
        zero_tokens = torch.zeros(
            (mzs.shape[0], self.n_decoder_frames),
            dtype=torch.long,
            device=self.device,
        )
        if self.self_cond_layers or return_intermediates:
            scores, intermediates = self.decoder.forward_self_conditioned(
                tokens=zero_tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
        else:
            scores = self.decoder(
                tokens=zero_tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
            intermediates = []

        if return_intermediates:
            return scores, seqs, intermediates
        return scores, seqs

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """

        pred, truth, intermediates = self._forward_step(
            batch, return_intermediates=True
        )

        if self.chimera:
            loss, aux = self._chimera_loss(pred, intermediates, batch)
        else:
            loss, aux = self._single_loss(pred, intermediates, truth)

        # The final layer's CTC loss is what gets logged as `*_CTCLoss`, so it
        # stays comparable with runs that predate self-conditioning and keeps
        # driving best-checkpoint selection. The combined objective below is
        # what the optimizer sees; logging that instead would mix in the
        # shallower layers' worse predictions and both compare wrongly against
        # earlier curves and pick checkpoints on the wrong quantity.
        self.log(
            f"{mode}_CTCLoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=truth.shape[0],
        )
        if aux is not None:
            loss = (1 - self.self_cond_weight) * loss + (
                self.self_cond_weight * aux
            )
            for key, value in (
                (f"{mode}_CTCLoss_intermediate", aux),
                (f"{mode}_CTCLoss_total", loss),
            ):
                self.log(
                    key,
                    value.detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=truth.shape[0],
                )
        return loss

    def _ctc_per_spectrum(
        self,
        scores: torch.Tensor,
        truth: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        The unnormalized CTC loss of each spectrum.

        Parameters
        ----------
        scores : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The frame-level scores to align against the targets.
        truth : torch.Tensor of shape (n_spectra, max_len)
            The padded target token indices.
        target_lengths : torch.Tensor of shape (n_spectra,)
            The number of real tokens in each target.

        Returns
        -------
        torch.Tensor of shape (n_spectra,)
            The loss of each spectrum, not divided by its target length.
        """
        log_probs = scores.log_softmax(-1).transpose(0, 1)  # (T, B, V)
        input_lengths = torch.full(
            (truth.shape[0],),
            log_probs.shape[0],
            dtype=torch.long,
            device=truth.device,
        )
        return self.ctc_loss_per_seq(
            log_probs, truth, input_lengths, target_lengths
        )

    def _warn_if_infeasible(
        self,
        truth: torch.Tensor,
        target_lengths: torch.Tensor,
        n_frames: int,
    ) -> None:
        """
        Warn once if any target needs more frames than it is given.

        CTC needs one frame per token plus a blank between repeated tokens.
        A longer peptide is unalignable and gets zero loss (``zero_infinity``),
        so it silently drops out of training.

        Parameters
        ----------
        truth : torch.Tensor of shape (n_spectra, max_len)
            The padded target token indices.
        target_lengths : torch.Tensor of shape (n_spectra,)
            The number of real tokens in each target.
        n_frames : int
            The frames available to align them to.
        """
        if self._ctc_infeasible_warned or truth.shape[1] < 2:
            return
        repeats = ((truth[:, 1:] == truth[:, :-1]) & (truth[:, 1:] != 0)).sum(
            dim=1
        )
        infeasible = target_lengths + repeats > n_frames
        if infeasible.any():
            self._ctc_infeasible_warned = True
            logger.warning(
                "%d peptide(s) in this batch need more CTC frames than the "
                "%d available and will contribute zero loss. Increase "
                "max_peptide_len to include them in training.",
                infeasible.sum().item(),
                n_frames,
            )

    def _single_loss(
        self,
        pred: torch.Tensor,
        intermediates: List[torch.Tensor],
        truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The CTC loss of one peptide spanning all of the decoder frames.

        Parameters
        ----------
        pred : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The final layer's frame-level scores.
        intermediates : List[torch.Tensor]
            The self-conditioning layers' own scores, if any.
        truth : torch.Tensor of shape (n_spectra, max_len)
            The padded target token indices.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            The final layer's loss, and the mean loss over the
            self-conditioning layers or None when there are none.
        """
        target_lengths = (truth != 0).sum(dim=1)
        self._warn_if_infeasible(truth, target_lengths, pred.shape[1])

        def reduce(scores):
            per_spectrum = self._ctc_per_spectrum(
                scores, truth, target_lengths
            )
            # Reproduces CTCLoss(reduction="mean"), which divides each
            # spectrum by its target length before averaging.
            return (per_spectrum / target_lengths.clamp(min=1)).mean()

        loss = reduce(pred)
        if not intermediates:
            return loss, None
        # Self-conditioned CTC: the same objective on each conditioning
        # layer's own prediction, so those layers are trained to say
        # something worth feeding forward.
        aux = torch.stack([reduce(scores) for scores in intermediates]).mean()
        return loss, aux

    def _chimera_loss(
        self,
        pred: torch.Tensor,
        intermediates: List[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The CTC loss of two peptides, one per slot of decoder frames.

        Nothing says which peptide belongs in which slot, so both
        assignments are scored and the cheaper one is kept. Only that
        assignment contributes gradient. The choice is made once, from the
        final layer, and reused for every self-conditioning layer: letting
        each layer pick its own would train them toward conflicting
        assignments, and a prediction fed forward under one assignment
        would be read by the next layer under another.

        A slot the annotation leaves empty contributes nothing, so the
        model is never trained to stay silent in it. See the comment on
        ``present_a`` below for why.

        Parameters
        ----------
        pred : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The final layer's frame-level scores.
        intermediates : List[torch.Tensor]
            The self-conditioning layers' own scores, if any.
        batch : Dict[str, torch.Tensor]
            The batch, holding both targets under ``seq`` and ``seq_2``.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            The final layer's loss, and the mean loss over the
            self-conditioning layers or None when there are none.
        """
        truth_a = batch["seq"]
        truth_b = batch["seq_2"].to(truth_a.device)
        len_a = (truth_a != 0).sum(dim=1)
        len_b = (truth_b != 0).sum(dim=1)

        slots = (
            slice(None, self.chimera_split),
            slice(self.chimera_split, None),
        )
        # Either peptide may land in either slot, so both have to fit the
        # smaller one. Slot A is the longer, by the global precursor frame.
        narrowest = pred.shape[1] - self.chimera_split
        for truth, lengths in ((truth_a, len_a), (truth_b, len_b)):
            self._warn_if_infeasible(truth, lengths, narrowest)

        # Both assignments share the same two targets, so they share this
        # denominator and it cannot affect which one wins.
        total_len = (len_a + len_b).clamp(min=1)

        # A slot whose target is empty contributes nothing. CTC scores an
        # empty target as the cost of emitting blanks across that slot,
        # which trains it silent, and an annotation naming one peptide is
        # not evidence that the spectrum held one: co-isolation is common
        # and the annotation records only what was identified. Penalising
        # the slot teaches the model to suppress the second peptides this
        # branch exists to find. Zeroing the term zeroes its gradient too,
        # so the slot is simply free.
        #
        # The minimum stays permutation-invariant. A single-peptide
        # spectrum scores min(ctc(slot A, pep), ctc(slot B, pep)), so the
        # labelled peptide goes to whichever slot explains it better and
        # the free prediction takes the other; neither slot is privileged.
        # Leaving the empty terms in would let them tip that choice, and
        # they are not even symmetric, since slot A carries one more frame.
        present_a = (len_a > 0).to(pred.dtype)
        present_b = (len_b > 0).to(pred.dtype)

        def assignments(scores):
            slot_a, slot_b = (scores[:, frames] for frames in slots)
            direct = (
                self._ctc_per_spectrum(slot_a, truth_a, len_a) * present_a
                + self._ctc_per_spectrum(slot_b, truth_b, len_b) * present_b
            )
            swapped = (
                self._ctc_per_spectrum(slot_a, truth_b, len_b) * present_b
                + self._ctc_per_spectrum(slot_b, truth_a, len_a) * present_a
            )
            return direct, swapped

        direct, swapped = assignments(pred)
        swap = swapped < direct
        loss = (torch.where(swap, swapped, direct) / total_len).mean()
        if not intermediates:
            return loss, None

        aux = []
        for scores in intermediates:
            direct, swapped = assignments(scores)
            aux.append((torch.where(swap, swapped, direct) / total_len).mean())
        return loss, torch.stack(aux).mean()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation
        # metrics from the predicted peptides.
        # FIXME: Remove work around when depthcharge reverse detokenization
        # bug is fixed.
        # peptides_true = self.tokenizer.detokenize(batch["seq"])
        logits, _ = self.forward(batch)
        # A chimeric spectrum contributes two pairs, so weight the logged
        # metrics by the number of spectra rather than the number of pairs.
        batch_size = batch["seq"].shape[0]
        if self.chimera:
            peptides_true, peptides_pred = self._chimera_eval_pairs(
                batch, logits
            )
        else:
            peptides_true = self._detokenize_targets(batch["seq"])
            peptides_pred = self._detokenize_decoded(logits)
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true, peptides_pred, self.tokenizer.residues
            )
        )

        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "pep_precision", pep_precision, **log_args, batch_size=batch_size
        )
        self.log(
            "aa_precision", aa_precision, **log_args, batch_size=batch_size
        )
        return loss

    def _detokenize_targets(self, truth: torch.Tensor) -> List[str]:
        """
        Turn padded target tokens back into peptide strings.

        Parameters
        ----------
        truth : torch.Tensor of shape (n_spectra, max_len)
            The padded target token indices. An all-padding row, which is
            how an absent second peptide is stored, gives ``""``.

        Returns
        -------
        List[str]
            One peptide string per spectrum.
        """
        # An all-padding row has to be held out: depthcharge cannot
        # detokenize one, just as it cannot tokenize an empty sequence.
        filled = [i for i, row in enumerate(truth) if (row != 0).any()]
        peptides = [""] * truth.shape[0]
        if not filled:
            return peptides
        decoded = self.tokenizer.detokenize(truth[filled], join=False)
        for i, peptide in zip(filled, decoded):
            peptides[i] = "".join(peptide)
        return peptides

    def _detokenize_decoded(self, logits: torch.Tensor) -> List[str]:
        """
        Greedily decode frames into peptide strings.

        Parameters
        ----------
        logits : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The frame-level scores to decode.

        Returns
        -------
        List[str]
            One peptide string per spectrum, ``""`` where nothing was
            decoded.
        """
        return [
            (
                "".join(
                    self.tokenizer.detokenize(
                        torch.tensor([tokens]), join=False
                    )[0]
                )
                if tokens
                else ""
            )
            for tokens in self._ctc_decode(logits)[0]
        ]

    def _chimera_eval_pairs(
        self, batch: Dict[str, torch.Tensor], logits: torch.Tensor
    ) -> Tuple[List[str], List[str]]:
        """
        Pair each spectrum's two truths with its two decoded slots.

        Nothing ties a slot to a particular peptide, so a spectrum whose
        slots came out in the other order is still correct. Both pairings
        are tried and the one matching more peptides exactly is kept.

        A spectrum annotated with one peptide has an empty second truth.
        Those pairs are left out rather than scored against the empty
        string, which is not a peptide and which the matcher cannot parse.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch, holding both targets under ``seq`` and ``seq_2``.
        logits : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The frame-level scores for every slot.

        Returns
        -------
        Tuple[List[str], List[str]]
            The true and predicted peptides, aligned pairwise.
        """
        truths = [
            self._detokenize_targets(batch["seq"]),
            self._detokenize_targets(batch["seq_2"]),
        ]
        preds = [
            self._detokenize_decoded(logits[:, : self.chimera_split]),
            self._detokenize_decoded(logits[:, self.chimera_split :]),
        ]

        peptides_true, peptides_pred = [], []
        for i in range(len(truths[0])):
            true_pair = [truths[0][i], truths[1][i]]
            pred_pair = [preds[0][i], preds[1][i]]
            direct = sum(t == p for t, p in zip(true_pair, pred_pair))
            swapped = sum(t == p for t, p in zip(true_pair, pred_pair[::-1]))
            if swapped > direct:
                pred_pair = pred_pair[::-1]
            for true_pep, pred_pep in zip(true_pair, pred_pair):
                if true_pep:
                    peptides_true.append(true_pep)
                    peptides_pred.append(pred_pep)
        return peptides_true, peptides_pred

    def _ctc_decode(
        self, logits: torch.Tensor
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Greedy CTC decoding of frame-level logits.

        Collapse repeated frame predictions, remove blanks, and truncate
        at the stop token. N-terminal modification tokens are only
        valid at the peptide's N-terminus; elsewhere they are re-decided
        from their frame with all N-terminal tokens masked out, and
        dropped if the re-decision yields a blank or stop token.

        Parameters
        ----------
        logits : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The frame-level amino acid scores.

        Returns
        -------
        sequences : List[List[int]]
            The decoded token indices for each spectrum.
        scores : List[List[float]]
            The confidence for each decoded token, taken as the maximum
            probability over the frames merged into that token.
        """
        probs = torch.softmax(logits, dim=-1)
        frame_confs, frame_tokens = probs.max(dim=-1)
        nterm_idx = set(self.nterm_idx.tolist())
        # Padding (0) is treated as non-emitting, like the blank.
        silent = (0, self.blank_token)

        sequences, scores = [], []
        for b, (tokens, confs) in enumerate(
            zip(frame_tokens.tolist(), frame_confs.tolist())
        ):
            seq, conf, frames, prev = [], [], [], self.blank_token
            for j, (token, prob) in enumerate(zip(tokens, confs)):
                if token not in silent:
                    if token != prev:
                        seq.append(token)
                        conf.append(prob)
                        frames.append(j)
                    elif prob > conf[-1]:
                        conf[-1] = prob
                        frames[-1] = j
                prev = token
            if self.stop_token in seq:
                stop = seq.index(self.stop_token)
                seq, conf, frames = seq[:stop], conf[:stop], frames[:stop]
            # With a reversed tokenizer the N-terminus is the last token.
            nterm_pos = len(seq) - 1 if self.tokenizer.reverse else 0
            fixed_seq, fixed_conf = [], []
            for i, (token, prob, frame) in enumerate(zip(seq, conf, frames)):
                if i != nterm_pos and token in nterm_idx:
                    masked = logits[b, frame].clone()
                    masked[self.nterm_idx] = -float("inf")
                    token = int(masked.argmax())
                    if token in silent or token == self.stop_token:
                        continue
                    prob = torch.softmax(masked, dim=0)[token].item()
                fixed_seq.append(token)
                fixed_conf.append(prob)
            sequences.append(fixed_seq)
            scores.append(fixed_conf)
        return sequences, scores

    def _precursor_candidates(
        self,
        precursor_mass: float,
        precursor_mz: Optional[float] = None,
        precursor_charge: Optional[int] = None,
    ) -> List[Tuple[Optional[int], float]]:
        """
        The (charge, neutral mass) pairs to accept for one spectrum.

        With ``charge_range`` unset this is the annotated charge alone,
        and ``precursor_mass`` is used as given. With a range set, the
        annotated charge is treated as a guess: the neutral mass is
        recomputed from the observed m/z at each charge in the range.

        Parameters
        ----------
        precursor_mass : float
            The neutral mass implied by the annotated charge.
        precursor_mz : Optional[float]
            The observed precursor m/z. Required to consider charges
            other than the annotated one; without it the annotated
            charge is the only candidate whatever ``charge_range`` says.
        precursor_charge : Optional[int]
            The annotated precursor charge, reported back for the
            no-range case so callers need no special case for it.

        Returns
        -------
        List[Tuple[Optional[int], float]]
            One (charge, neutral mass) pair per candidate charge.
        """
        if self.charge_range is None or precursor_mz is None:
            return [(precursor_charge, precursor_mass)]
        lo, hi = self.charge_range
        return [
            (z, (precursor_mz - PROTON_MASS) * z) for z in range(lo, hi + 1)
        ]

    def _residue_mass_windows(
        self,
        candidates: Sequence[Tuple[Optional[int], float]],
        guard: float = 0.0,
    ) -> List[Tuple[Optional[int], float, float]]:
        """
        Acceptable total residue mass windows for a precursor.

        One window per (candidate charge, isotope error) pair, each
        spanning the precursor mass tolerance (in ppm), expressed in
        residue-sum space (i.e. with the water mass removed).

        Parameters
        ----------
        candidates : Sequence[Tuple[Optional[int], float]]
            The (charge, neutral mass) pairs to accept, as returned by
            ``_precursor_candidates``.
        guard : float
            Extra widening (in Da) applied to both window edges.

        Returns
        -------
        List[Tuple[Optional[int], float, float]]
            The charge and the (lower, upper) bounds of each window.
        """
        return [
            (charge, center - tol - guard, center + tol + guard)
            for charge, precursor_mass in candidates
            for tol in [self.precursor_mass_tol * precursor_mass / 1e6]
            for iso in range(
                self.isotope_error_range[0], self.isotope_error_range[1] + 1
            )
            for center in [precursor_mass - iso * ISOTOPE_SPACING - H2O_MASS]
        ]

    def _matching_window(
        self,
        tokens: List[int],
        precursor_mass: float,
        precursor_mz: Optional[float] = None,
        precursor_charge: Optional[int] = None,
    ) -> Optional[Tuple[Optional[int], float, float]]:
        """
        The mass window a peptide falls in, if it falls in any.

        Returned rather than a bare charge because the charge is itself
        None when ``charge_range`` is unset and no annotated charge was
        supplied, which a caller must not confuse with "no match".

        Parameters
        ----------
        tokens : List[int]
            The decoded token indices.
        precursor_mass : float
            The neutral mass implied by the annotated charge.
        precursor_mz : Optional[float]
            The observed precursor m/z, needed to consider charges
            other than the annotated one.
        precursor_charge : Optional[int]
            The annotated precursor charge.

        Returns
        -------
        Optional[Tuple[Optional[int], float, float]]
            The charge and bounds of the first window the peptide's
            residue mass falls in, or None if it falls in none. Windows
            are tried in ascending charge order, so a peptide that fits
            at more than one charge is assigned the lowest.
        """
        mass = (
            self.token_masses[torch.tensor(tokens, dtype=torch.long)]
            .sum()
            .item()
        )
        candidates = self._precursor_candidates(
            precursor_mass, precursor_mz, precursor_charge
        )
        for window in self._residue_mass_windows(candidates):
            if window[1] <= mass <= window[2]:
                return window
        return None

    def _fits_precursor_mass(
        self,
        tokens: List[int],
        precursor_mass: float,
        precursor_mz: Optional[float] = None,
        precursor_charge: Optional[int] = None,
    ) -> bool:
        """
        Check whether a peptide matches the precursor mass.

        Parameters
        ----------
        tokens : List[int]
            The decoded token indices.
        precursor_mass : float
            The neutral mass implied by the annotated charge.
        precursor_mz : Optional[float]
            The observed precursor m/z, needed to consider charges
            other than the annotated one.
        precursor_charge : Optional[int]
            The annotated precursor charge.

        Returns
        -------
        bool
            True if the total residue mass falls within the precursor
            mass tolerance for any allowed isotope error, at any
            allowed charge.
        """
        return (
            self._matching_window(
                tokens, precursor_mass, precursor_mz, precursor_charge
            )
            is not None
        )

    def _pmc_decode(
        self,
        logits: torch.Tensor,
        precursor_mass: float,
        precursor_mz: Optional[float] = None,
        precursor_charge: Optional[int] = None,
    ) -> Optional[Tuple[List[int], List[float], Optional[int]]]:
        """
        Precise mass control (PMC) decoding for a single spectrum.

        Knapsack-like dynamic programming over the CTC lattice (after
        PrimeNovo): find the highest-probability CTC path whose
        collapsed peptide's total residue mass matches the precursor
        mass within tolerance, considering all allowed isotope errors
        and, when ``charge_range`` is set, all allowed charges.
        The DP state is (discretized emitted mass, last path symbol);
        the last symbol is needed to apply the CTC collapse rules
        (a repeated symbol without an intervening blank does not emit).

        Each state also carries the EXACT residue mass of the best path
        reaching it, and that is what the readout tests. The bins only
        partition the states. Rounding each residue onto the grid and
        testing the bin instead would let error accumulate over the
        emissions until the path the search commits to is one the exact
        check at the end rejects, losing the spectrum even though a
        slightly less probable path would have matched. Carrying the
        mass makes the search and the acceptance the same predicate, so
        that cannot happen (PrimeNovo's `mass_con.py` does the same).

        Stop and padding are excluded from the search, as is any
        non-terminal residue of zero or negative mass (it could repeat
        without bound, which a fixed axis cannot hold). N-terminal
        modifications may only be emitted as the peptide's first
        residue, and that is also what bounds the axis below zero: only
        they may carry a negative mass, so at most one such step is ever
        taken. As an approximation for speed, a token takes part in a
        frame only where its probability exceeds ``PMC_MIN_EMIT_PROB``:
        it cannot be emitted there, and a run of it cannot span that
        frame either, since the state is dropped rather than carried.
        Frames with no such token reduce to a blank-only update, which
        is most of them.

        The mass grid is discretized at ``PMC_RESOLUTION``, coarsened
        for heavy precursors so that the backtracking table stays under
        ``PMC_MAX_POINTER_BYTES``. Since the bins only separate states, a
        coarser grid costs recall rather than correctness: near-equal
        masses come to share a state, and only the more probable of them
        survives. Bin ``zero_bin`` holds zero mass, with the bins below
        it reserved for the negative excursion. One axis spans every
        candidate window, so a ``charge_range`` whose upper end is well
        above the annotated charge stretches the axis in proportion and
        can trip that coarsening on spectra where the annotated charge
        alone would not have.

        Parameters
        ----------
        logits : torch.Tensor of shape (n_frames, n_tokens)
            The frame-level amino acid scores for one spectrum.
        precursor_mass : float
            The neutral mass implied by the annotated charge.
        precursor_mz : Optional[float]
            The observed precursor m/z, needed to consider charges
            other than the annotated one.
        precursor_charge : Optional[int]
            The annotated precursor charge.

        Returns
        -------
        Optional[Tuple[List[int], List[float], Optional[int]]]
            The decoded token indices, the per-token confidences, and
            the charge whose window the peptide matched, or None if no
            mass-matching path exists (or the search was skipped because
            the DP table would be too large).
        """
        device = logits.device
        n_frames, vocab = logits.shape
        # A reversed tokenizer emits the peptide C-terminus first, which
        # would put an N-terminal modification last. Decoding the frames
        # back to front puts it first in either case, so the "N-terminal
        # tokens open the peptide" rule below is the only one needed;
        # the emitted tokens are flipped back before returning.
        if self.tokenizer.reverse:
            logits = logits.flip(0)

        # Mass discretization. The pointer table is the memory
        # bottleneck, so pick the finest resolution whose table fits in
        # PMC_MAX_POINTER_BYTES rather than giving up on mass control
        # for heavy precursors. The guard widens the AXIS, not the
        # acceptance test: a path whose exact mass is in tolerance can
        # still have its bin carried off the window edge by accumulated
        # rounding, and needs somewhere to live when it is. That error
        # grows with the resolution and the number of emissions, and the
        # guard reduces to PMC_MASS_GUARD at the default resolution.
        candidates = self._precursor_candidates(
            precursor_mass, precursor_mz, precursor_charge
        )
        base = self._residue_mass_windows(candidates, PMC_MASS_GUARD)
        hi_base = max(hi for _, _, hi in base)
        if hi_base <= 0:
            return None
        # An unbounded window constrains nothing: precursor_mass_tol is
        # "inf", which is how PMC is turned off. Bail out rather than build
        # a mass axis of infinite extent, whose resolution would come out
        # inf and whose bin count would be NaN. Only an empty greedy
        # peptide reaches here under an infinite tolerance, since a
        # non-empty one already "fits" and predict_step skips the search.
        if not np.isfinite(hi_base):
            return None
        max_bins = max(4, PMC_MAX_POINTER_BYTES // (n_frames * vocab))
        resolution = max(PMC_RESOLUTION, hi_base / (max_bins - 2))
        guard = max(PMC_MASS_GUARD, float(np.sqrt(n_frames)) * resolution)
        axis_windows = self._residue_mass_windows(candidates, guard)
        hi_max = max(hi for _, _, hi in axis_windows)
        # What the readout accepts: the true windows, unguarded. The
        # states carry exact masses, so this is the same predicate
        # `_matching_window` applies to the finished peptide.
        accept_windows = self._residue_mass_windows(candidates)
        # Negative-mass tokens (the ammonia-loss N-terminal modification)
        # push the running total below zero, so the axis starts below it.
        # Only N-terminal modifications may be negative and only one may be
        # emitted, so one token's worth of headroom is enough.
        neg_mass = min(
            (self.token_masses[c].item() for c in self.nterm_idx.tolist()),
            default=0.0,
        )
        zero_bin = int(np.ceil(max(0.0, -neg_mass) / resolution)) + 1
        n_bins = zero_bin + int(hi_max / resolution) + 2
        if n_bins > max_bins:
            # The widened guard pushed the table back over budget. The
            # offset bins have to come out of the budget too, so solve for
            # the resolution with them already reserved.
            resolution = hi_max / max(1, max_bins - zero_bin - 2)
            zero_bin = int(np.ceil(max(0.0, -neg_mass) / resolution)) + 1
            n_bins = zero_bin + int(hi_max / resolution) + 2
        if resolution > PMC_RESOLUTION and not self._pmc_size_warned:
            self._pmc_size_warned = True
            logger.warning(
                "Coarsening precise mass control decoding to %.4f Da for "
                "large precursor masses to keep the DP table under %d "
                "bytes.",
                resolution,
                PMC_MAX_POINTER_BYTES,
            )

        # Emittable tokens and their discretized masses. Padding, the
        # blank, and the stop token do not emit. N-terminal modifications
        # are emittable but only as the first residue (see `nterm_only`),
        # which is also what bounds the negative headroom: a negative mass
        # is allowed only for those, so at most one can be emitted.
        excluded = {0, self.blank_token, self.stop_token}
        nterm = set(self.nterm_idx.tolist())
        emit_tokens, emit_deltas, emit_nterm = [], [], []
        for c in range(min(vocab, self.token_masses.shape[0])):
            mass = self.token_masses[c].item()
            if c in excluded or mass > hi_max:
                continue
            if mass <= 0 and c not in nterm:
                # A non-terminal residue of zero or negative mass could
                # repeat without bound, which the fixed axis cannot hold.
                continue
            emit_tokens.append(c)
            emit_deltas.append(round(mass / resolution))
            emit_nterm.append(c in nterm)
        if not emit_tokens:
            return None
        deltas = dict(zip(emit_tokens, emit_deltas))
        emit_idx = torch.tensor(emit_tokens, device=device)
        emit_i8 = emit_idx.unsqueeze(1).to(torch.int8)
        delta_idx = torch.tensor(emit_deltas, device=device)
        # The unrounded masses, added to the running total alongside the
        # rounded ones that move the state along the axis.
        emit_exact = self.token_masses[emit_idx]

        # Per-token mass shifts as gather indices over the mass axis:
        # src_bins[e, m] = m - delta_e. A negative delta moves the source
        # ABOVE the target, so both ends of the axis have to be checked.
        bins = torch.arange(n_bins, device=device)
        raw_src = bins.unsqueeze(0) - delta_idx.unsqueeze(1)
        valid = (raw_src >= 0) & (raw_src < n_bins)
        src_bins = raw_src.clamp(0, n_bins - 1)
        # An N-terminal modification may only open the peptide. "Nothing
        # emitted yet" is the zero-mass bin, so requiring the source to be
        # `zero_bin` means requiring the target to be zero_bin + delta.
        nterm_only = torch.tensor(emit_nterm, device=device).unsqueeze(1)
        valid &= ~nterm_only | (
            bins.unsqueeze(0) == zero_bin + delta_idx.unsqueeze(1)
        )

        log_probs = logits.log_softmax(-1)
        neg_inf = float("-inf")
        blank = self.blank_token
        # score[m, c]: best log-probability of any path prefix with
        # discretized emitted mass m and last symbol c.
        score = torch.full((n_bins, vocab), neg_inf, device=device)
        score[zero_bin, blank] = 0.0
        # state_mass[m, c]: the exact residue mass of that same best
        # prefix, accumulated from the unrounded token masses. NaN marks
        # a state no path has reached; it compares false against every
        # window, so an unreachable state can never be read out.
        state_mass = torch.full(
            (n_bins, vocab), float("nan"), dtype=torch.float64, device=device
        )
        state_mass[zero_bin, blank] = 0.0
        # pointers[t, m, c]: last symbol of the predecessor state; a
        # pointer equal to c itself encodes a repeat (no new emission).
        # Kept on the model device; backtracking reads single entries.
        # int8 holds the symbol index, so the alphabet has to fit in it:
        # above 127 the cast would wrap and backtracking would follow the
        # wrong symbols without any error.
        if vocab > 127:
            raise ValueError(
                f"PMC decoding stores predecessors as int8, so it supports "
                f"at most 127 tokens, but the vocabulary has {vocab}"
            )
        pointers = torch.empty(
            (n_frames, n_bins, vocab), dtype=torch.int8, device=device
        )
        # Tokens below PMC_MIN_EMIT_PROB at a frame drop out of that
        # frame entirely: they cannot be emitted, and an ongoing run of
        # one cannot continue through it, because the state is left at
        # -inf rather than carried forward. Frames with no candidate at
        # all reduce to a blank-only update, which is the usual case and
        # the reason this is affordable: measured 7 of 101 frames doing
        # emission work on a typical spectrum, 120 ms against 1,378 ms
        # with the threshold removed.
        min_lp = float(np.log(PMC_MIN_EMIT_PROB))
        candidates = log_probs[:, emit_idx] > min_lp  # (n_frames, E)
        for t in range(n_frames):
            lp = log_probs[t]
            sel = candidates[t].nonzero(as_tuple=True)[0]
            new_score = torch.full_like(score, neg_inf)
            new_ptr = torch.full(
                (n_bins, vocab), -1, dtype=torch.int8, device=device
            )
            new_mass = torch.full_like(state_mass, float("nan"))
            if sel.numel() == 0:
                # Blank-only frame: every state transitions to blank.
                best, best_arg = score.max(dim=1)
                new_score[:, blank] = best + lp[blank]
                new_ptr[:, blank] = best_arg.to(torch.int8)
                # A blank emits nothing, so the mass rides along.
                new_mass[:, blank] = state_mass.gather(
                    1, best_arg.unsqueeze(1)
                ).squeeze(1)
                score = new_score
                state_mass = new_mass
                pointers[t] = new_ptr
                continue
            top2 = score.topk(2, dim=1)
            best, best_arg = top2.values[:, 0], top2.indices[:, 0]
            second, second_arg = top2.values[:, 1], top2.indices[:, 1]
            # The blank column's predecessor mass; the emission columns
            # read theirs off `emit_arg` below, which already names the
            # predecessor the score chose.
            mass_best = state_mass.gather(1, best_arg.unsqueeze(1)).squeeze(1)
            sel_tokens = emit_idx[sel]
            # Best predecessor excluding each candidate token itself,
            # shifted by that token's mass: a new emission of token e at
            # mass m extends the best path at mass m - delta_e whose
            # last symbol differs from e.
            is_self = best_arg.unsqueeze(0) == sel_tokens.unsqueeze(1)
            excl_val = torch.where(
                is_self, second.unsqueeze(0), best.unsqueeze(0)
            )
            excl_arg = torch.where(
                is_self, second_arg.unsqueeze(0), best_arg.unsqueeze(0)
            )
            emit_val = excl_val.gather(1, src_bins[sel]) + lp[
                sel_tokens
            ].unsqueeze(1)
            emit_val = emit_val.masked_fill(~valid[sel], neg_inf)
            emit_arg = excl_arg.gather(1, src_bins[sel])
            # The exact mass follows the same predecessor the score does
            # -- state (src_bins, emit_arg) -- gaining the unrounded
            # residue mass rather than its bin.
            emit_mass = state_mass.view(-1)[
                src_bins[sel] * vocab + emit_arg
            ] + emit_exact[sel].unsqueeze(1)
            # Continue the current run: no new emission.
            repeat = score[:, sel_tokens].T + lp[sel_tokens].unsqueeze(1)
            repeat_mass = state_mass[:, sel_tokens].T
            use_emit = emit_val > repeat
            emit_score = torch.where(use_emit, emit_val, repeat)
            emit_state_mass = torch.where(use_emit, emit_mass, repeat_mass)
            emit_ptr = torch.where(
                use_emit,
                emit_arg.to(torch.int8),
                emit_i8[sel].expand_as(use_emit),
            )
            # Blank keeps the mass and may follow any symbol.
            new_score[:, blank] = best + lp[blank]
            new_ptr[:, blank] = best_arg.to(torch.int8)
            new_mass[:, blank] = mass_best
            new_score[:, sel_tokens] = emit_score.T
            new_ptr[:, sel_tokens] = emit_ptr.T
            new_mass[:, sel_tokens] = emit_state_mass.T
            score = new_score
            state_mass = new_mass
            pointers[t] = new_ptr

        # Read out the best final state whose EXACT mass is in tolerance.
        # Testing the carried mass rather than the bin the state sits in
        # is what keeps accumulated rounding out of the decision: a state
        # is a candidate only if the peptide it stands for really does
        # match the precursor. The search can no longer commit to a path
        # that the check below then throws away, which used to lose the
        # spectrum outright even when a slightly less probable path in a
        # neighbouring bin would have matched.
        allowed = torch.zeros_like(score, dtype=torch.bool)
        for _, lo, hi in accept_windows:
            allowed |= (state_mass >= lo) & (state_mass <= hi)
        score = score.masked_fill(~allowed, neg_inf)
        best_val, flat_idx = score.flatten().max(dim=0)
        if not torch.isfinite(best_val):
            return None
        m = int(flat_idx.item()) // vocab
        c = int(flat_idx.item()) % vocab

        # Backtrack, aggregating each emission's confidence as the
        # maximum probability over its merged frames.
        probs = log_probs.exp().cpu()
        emissions = []
        run_conf = None
        for t in reversed(range(n_frames)):
            prev = int(pointers[t, m, c])
            if c != blank:
                prob = probs[t, c].item()
                run_conf = prob if run_conf is None else max(run_conf, prob)
                if prev != c:
                    # Frame t started this emission of c.
                    emissions.append((c, run_conf))
                    run_conf = None
                    m -= deltas[c]
            c = prev

        tokens = [c for c, _ in reversed(emissions)]
        confs = [s for _, s in reversed(emissions)]
        if self.tokenizer.reverse:
            # Undo the frame flip so the caller still receives the
            # tokens in the tokenizer's own order.
            tokens.reverse()
            confs.reverse()
        # Which charge the path matched. The DP ran on one axis covering
        # every candidate window, so the window has to be recovered here.
        # This re-checks the mass, but on a path already selected for
        # having an exact mass in tolerance it now agrees by
        # construction, up to the last bit of the summation order.
        if not tokens:
            return None
        window = self._matching_window(
            tokens, precursor_mass, precursor_mz, precursor_charge
        )
        if window is None:
            return None
        return tokens, confs, window[0]

    def predict_step(
        self, batch: Dict[str, torch.Tensor], *args
    ) -> List[psm.PepSpecMatch]:
        """
        A single prediction step (greedy CTC decoding).

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, containing keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``,
            ``precursor_charge``, plus metadata like ``peak_file`` and
            ``scan_id``.

        Returns
        -------
        predictions : List[psm.PepSpecMatch]
            Predicted PSMs for the given batch of spectra.
        """
        logits, _ = self._forward_step(batch)
        _, _, precursors, _ = self._process_batch(batch)

        if self.chimera:
            # One peptide per slot of frames. Each slot is a self-contained
            # single-peptide problem over its own frames, so the mass
            # constraint applies to it unchanged, and each slot finds its
            # own charge: two co-isolated precursors rarely share one.
            slots = [
                self._decode_frames(
                    logits[:, : self.chimera_split], precursors
                ),
                self._decode_frames(
                    logits[:, self.chimera_split :], precursors
                ),
            ]
        else:
            slots = [self._decode_frames(logits, precursors)]

        predictions = []
        for i, prec_mz in enumerate(batch["precursor_mz"]):
            spectrum_id = (batch["peak_file"][i], batch["scan_id"][i])
            exp_mz = float(prec_mz.item())
            matches = []
            for sequences, scores, charges in slots:
                match = self._build_psm(
                    sequences[i], scores[i], charges[i], spectrum_id, exp_mz
                )
                if match is not None:
                    matches.append(match)
            if self.chimera:
                matches = self._select_chimera(matches)
            predictions.extend(matches)

        return predictions

    def _decode_frames(
        self, logits: torch.Tensor, precursors: torch.Tensor
    ) -> Tuple[List[List[int]], List[List[float]], List[int]]:
        """
        Decode one peptide per spectrum from a range of frames.

        Greedy CTC decoding, then precise mass control: when the greedy
        peptide does not match the precursor mass, search for the best CTC
        path that does.

        Parameters
        ----------
        logits : torch.Tensor of shape (n_spectra, n_frames, n_tokens)
            The frame-level scores to decode. This is every frame for
            single-peptide sequencing, and one slot's frames for chimeric.
        precursors : torch.Tensor of shape (n_spectra, 3)
            The precursor neutral mass, charge, and m/z.

        Returns
        -------
        Tuple[List[List[int]], List[List[float]], List[int]]
            The decoded tokens, their confidences, and the charge each
            peptide was accepted under. With ``charge_range`` unset that
            charge is always the annotated one; with a range set it is the
            charge whose window the peptide matched, which is what
            ``calc_mz`` has to be computed from downstream.
        """
        sequences, scores = self._ctc_decode(logits)
        charges = [int(z) for z in precursors[:, 1].tolist()]
        for i, tokens in enumerate(sequences):
            precursor_mass = precursors[i, 0].item()
            precursor_mz = precursors[i, 2].item()
            annotated = charges[i]
            if tokens:
                window = self._matching_window(
                    tokens, precursor_mass, precursor_mz, annotated
                )
                if window is not None:
                    charges[i] = window[0]
                    continue
            pmc = self._pmc_decode(
                logits[i], precursor_mass, precursor_mz, annotated
            )
            if pmc is not None:
                sequences[i], scores[i], charges[i] = pmc
            elif self.chimera and tokens:
                # No window accepted this peptide, so no charge was chosen
                # for it. Reporting the annotated charge would be wrong
                # more often than not: it belongs to whichever precursor
                # the instrument picked, and this is the other one.
                charges[i] = self._best_effort_charge(
                    tokens, precursor_mass, precursor_mz, annotated
                )
        return sequences, scores, charges

    def _best_effort_charge(
        self,
        tokens: List[int],
        precursor_mass: float,
        precursor_mz: Optional[float],
        precursor_charge: Optional[int],
    ) -> int:
        """
        The charge that comes closest to explaining a peptide's mass.

        Used only when no candidate window accepted the peptide, so that a
        chimeric sub-peptide still gets the most plausible charge rather
        than the annotated one. Unlike ``_matching_window`` this always
        returns a charge, however poor the match.

        The candidates and the mass centers are the same ones
        ``_matching_window`` tests, so this picks the window the peptide
        came nearest to missing. With ``charge_range`` unset there is only
        one candidate and the annotated charge comes straight back.

        Parameters
        ----------
        tokens : List[int]
            The decoded token indices.
        precursor_mass : float
            The neutral mass implied by the annotated charge.
        precursor_mz : Optional[float]
            The observed precursor m/z, needed to consider other charges.
        precursor_charge : Optional[int]
            The annotated precursor charge, returned when there is nothing
            to choose between.

        Returns
        -------
        int
            The best-matching charge.
        """
        residue_mass = (
            self.token_masses[torch.tensor(tokens, dtype=torch.long)]
            .sum()
            .item()
        )
        best_charge, best_error = precursor_charge, float("inf")
        for charge, mass in self._precursor_candidates(
            precursor_mass, precursor_mz, precursor_charge
        ):
            for iso in range(
                self.isotope_error_range[0], self.isotope_error_range[1] + 1
            ):
                center = mass - iso * ISOTOPE_SPACING - H2O_MASS
                # Relative to the candidate's own mass, which is how
                # `_residue_mass_windows` scales its ppm tolerance.
                error = abs(residue_mass - center) / max(abs(mass), 1e-9)
                if error < best_error:
                    best_charge, best_error = charge, error
        return int(best_charge if best_charge is not None else 1)

    def _build_psm(
        self,
        tokens: List[int],
        confs: List[float],
        charge: int,
        spectrum_id: Tuple[str, str],
        exp_mz: float,
    ) -> Optional[psm.PepSpecMatch]:
        """
        Assemble one PSM from decoded tokens.

        Parameters
        ----------
        tokens : List[int]
            The decoded token indices.
        confs : List[float]
            The confidence of each decoded token.
        charge : int
            The charge the peptide was accepted under.
        spectrum_id : Tuple[str, str]
            The peak file and scan identifier.
        exp_mz : float
            The observed precursor m/z.

        Returns
        -------
        Optional[psm.PepSpecMatch]
            The PSM, or None if nothing was decoded.
        """
        if not tokens:
            return None

        peptide = "".join(
            self.tokenizer.detokenize(torch.tensor([tokens]), join=False)[0]
        )
        aa_scores = np.array(confs)
        if self.tokenizer.reverse:
            aa_scores = aa_scores[::-1]

        return psm.PepSpecMatch(
            sequence=peptide,
            spectrum_id=spectrum_id,
            peptide_score=float(aa_scores.mean()),
            charge=int(charge),
            calc_mz=np.nan,
            exp_mz=exp_mz,
            aa_scores=aa_scores,
        )

    def _select_chimera(
        self, matches: List[psm.PepSpecMatch]
    ) -> List[psm.PepSpecMatch]:
        """
        Choose which of a spectrum's slots to report.

        A slot that decoded nothing has already been dropped. Of the rest,
        discard peptides too short to report and those whose N-terminal
        modification sits past the start, which is not valid ProForma.
        Both slots can converge on the same peptide, so keep the
        higher-scoring copy of any duplicate.

        Parameters
        ----------
        matches : List[psm.PepSpecMatch]
            The PSMs decoded from one spectrum, at most one per slot.

        Returns
        -------
        List[psm.PepSpecMatch]
            The PSMs to report, at most one per slot.
        """
        kept, seen = [], set()
        for match in sorted(
            matches, key=lambda m: m.peptide_score, reverse=True
        ):
            # Checked before the length, which parses the sequence: a
            # misplaced N-terminal modification is not valid ProForma and
            # the parser raises on it.
            if self._has_internal_nterm_mod(match.sequence):
                continue
            if self._residue_len(match.sequence) < self.min_peptide_len:
                continue
            if match.sequence in seen:
                continue
            seen.add(match.sequence)
            kept.append(match)
        return kept

    def _residue_len(self, sequence: str) -> int:
        """
        The number of amino acid residues in a ProForma sequence.

        Counts residue tokens only; N-terminal modification tokens, which
        end in ``-``, are excluded.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        int
            The residue count.
        """
        return sum(
            1
            for tok in self.tokenizer.split(sequence)
            if not tok.endswith("-")
        )

    def _has_internal_nterm_mod(self, sequence: str) -> bool:
        """
        Whether an N-terminal modification appears past the start.

        A single leading N-terminal modification is valid; any further
        occurrence makes the ProForma invalid, as in ``PEP[Acetyl]-TIDE``.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        bool
            True if a modification sits somewhere it cannot be.
        """
        rest = sequence
        for mod in self.n_term:
            if rest.startswith(mod):
                rest = rest[len(mod) :]
                break
        return any(mod in rest for mod in self.n_term)

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        if "train_CTCLoss" in self.trainer.callback_metrics:
            train_loss = (
                self.trainer.callback_metrics["train_CTCLoss"].detach().item()
            )
        else:
            train_loss = np.nan
        metrics = {"step": self.trainer.global_step, "train": train_loss}
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CTCLoss"].detach().item(),
        }

        if self.calculate_precision:
            metrics["valid_aa_precision"] = (
                callback_metrics["aa_precision"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["pep_precision"].detach().item()
            )
        self._history.append(metrics)
        self._log_history()

    def on_predict_batch_end(
        self, outputs: List[psm.PepSpecMatch], *args
    ) -> None:
        """
        Write the predicted PSMs to the output file.

        Parameters
        ----------
        outputs : List[psm.PepSpecMatch]
            The predicted PSMs for the processed batch.
        """
        if self.out_writer is None:
            return

        for spec_match in outputs:
            if not spec_match.sequence:
                continue

            # N terminal scores should be combined with first token
            if len(spec_match.aa_scores) >= 2 and any(
                spec_match.sequence.startswith(mod) for mod in self.n_term
            ):
                spec_match.aa_scores[1] *= spec_match.aa_scores[0]
                spec_match.aa_scores = spec_match.aa_scores[1:]

            # Compute the precursor m/z of the predicted peptide.
            spec_match.calc_mz = self.tokenizer.calculate_precursor_ions(
                spec_match.sequence, torch.tensor(spec_match.charge)
            ).item()

            self.out_writer.psms.append(spec_match)

    def on_train_start(self):
        """Log optimizer settings."""
        self.log("hp/optimizer_warmup_iters", self.warmup_iters)
        self.log(
            "hp/optimizer_cosine_schedule_period_iters",
            self.cosine_schedule_period_iters,
        )

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tValid loss\t"
            if self.calculate_precision:
                header += "Peptide precision\tAA precision"

            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
            ]

            if self.calculate_precision:
                msg += "\t%.6f\t%.6f"
                vals += [
                    metrics.get("valid_pep_precision", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                ]

            logger.info(msg, *vals)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], Dict[str, Any]]:
        """
        Initialize the optimizer.

        Hidden weight matrices are optimized with Muon; embeddings, the
        output head, and all vector/scalar parameters use an auxiliary
        AdamW group, per the Muon usage guidance. A single cosine
        learning rate scheduler rescales both groups' base learning
        rates by the same warmup/decay factor.

        Returns
        -------
        Tuple[List[torch.optim.Optimizer], Dict[str, Any]]
            The initialized optimizer and its learning rate scheduler.
        """
        aux_ids = set()
        for module in self.modules():
            if isinstance(module, torch.nn.Embedding):
                aux_ids.update(id(p) for p in module.parameters())
        aux_ids.update(id(p) for p in self.decoder.final.parameters())
        muon_params, aux_params = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and id(p) not in aux_ids:
                muon_params.append(p)
            else:
                aux_params.append(p)
        optimizer = MuonWithAuxAdamW(
            [
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=self.muon_lr,
                    momentum=self.muon_momentum,
                    weight_decay=self.opt_kwargs.get("weight_decay", 0.0),
                ),
                dict(params=aux_params, use_muon=False, **self.opt_kwargs),
            ]
        )
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class DbSpec2Pep(Spec2Pep):
    """
    Subclass of Spec2Pep for the use of Casanovo as an MS/MS database
    search score function.

    Uses teacher forcing to 'query' Casanovo to score a peptide-spectrum
    pair. Note that this does *not* involve training, but rather that
    teacher forcing is used for predicting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        The forward step.

        If the encoder output is already present in the batch, it is used
        directly by the decoder. Otherwise, the full forward pass including
        the encoder is performed.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset. It must contain ``seq``.
            For a full forward pass, it also needs ``mz_array``,
            ``intensity_array``, ``precursor_mz``, and ``precursor_charge``.
            Alternatively, it can contain precomputed encoder outputs:
            ``memory``, ``mem_masks``, and ``precursors``.

        Returns
        -------
        scores : torch.Tensor of shape (B, length, n_amino_acids)
            The individual amino acid scores for each prediction,
            converted to probabilities using a softmax.
        tokens : torch.Tensor of shape (B, length)
            The ground truth tokens for each spectrum.

        Notes
        -----
        Here ``B`` denotes the number of peptide–spectrum pairs in the
        current candidate batch (or the number of spectra for a plain
        forward pass).
        """
        if (
            "memory" in batch
            and "mem_masks" in batch
            and "precursors" in batch
        ):
            memories, mem_masks = batch["memory"], batch["mem_masks"]
            precursors = batch["precursors"]
            tokens = batch["seq"]
            logits = self.decoder(
                tokens=tokens,
                memory=memories,
                memory_key_padding_mask=mem_masks,
                precursors=precursors,
            )
            probs = self.softmax(logits)
            return probs, tokens
        else:
            pred, truth = self._forward_step(batch)
            pred = self.softmax(pred)
            return pred, truth

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
    ) -> List[ms_io.PepSpecMatch]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.

        Returns
        -------
        predictions: List[ms_io.PepSpecMatch]
            The predicted PSMs for the processed batch.
        """
        predictions = collections.defaultdict(list)

        with torch.inference_mode():
            # Pre-compute encoder outputs for the entire batch.
            mzs, intensities, precursors_all, _ = self._process_batch(batch)
            memories, mem_masks = self.encoder(mzs, intensities)
            enc_cache = {
                "memory": memories,
                "mem_masks": mem_masks,
                "precursors_all": precursors_all,
            }

            for psm_batch in self._psm_batches(batch, enc_cache=enc_cache):
                pred_logits, truth = self.forward(psm_batch)
                peptide_scores, aa_scores_all = _calc_match_score(
                    pred_logits, truth
                )

                for (
                    filename,
                    scan,
                    precursor_charge,
                    precursor_mz,
                    peptide,
                    peptide_score,
                    curr_aa_scores,
                ) in zip(
                    psm_batch["peak_file"],
                    psm_batch["scan_id"],
                    psm_batch["precursor_charge"],
                    psm_batch["precursor_mz"],
                    psm_batch["original_seq_str"],
                    peptide_scores,
                    aa_scores_all,
                ):
                    # Omit stop token from reported AA scores.
                    curr_aa_scores = curr_aa_scores[:-1]
                    if self.tokenizer.reverse:
                        curr_aa_scores = curr_aa_scores[::-1]

                    spectrum_id = (filename, scan)
                    predictions[spectrum_id].append(
                        psm.PepSpecMatch(
                            sequence=peptide,
                            spectrum_id=spectrum_id,
                            peptide_score=peptide_score,
                            charge=int(precursor_charge),
                            calc_mz=np.nan,
                            exp_mz=precursor_mz.item(),
                            aa_scores=curr_aa_scores,
                        )
                    )

        # Filter the top-scoring prediction for each spectrum.
        predictions = list(
            itertools.chain.from_iterable(
                sorted(
                    spectrum_predictions,
                    key=lambda p: p.peptide_score,
                    reverse=True,
                )[: self.top_match]
                for spectrum_predictions in predictions.values()
            )
        )

        # Determine the parent proteins only for the retained PSMs.
        for pred in predictions:
            pred.protein = self.protein_database.get_associated_protein(
                pred.sequence
            )

        return predictions

    def _psm_batches(
        self,
        batch: Dict[str, torch.Tensor],
        enc_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generates batches of candidate database PSMs.

        PSM batches consist of repeated spectrum information for each
        candidate peptide to be scored against each spectrum.
        This method ensures that the batches provided to the model
        are of a consistent size.

        FIXME: Move this logic to a subclassed DataLoader.
         This would also allow correctly setting the batch size (now the
         final batch will be (much) smaller depending on how many
         spectra remain).

        TODO: The batch creation and generation could potentially be
         improved using a producer-consumer pattern.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data.
        enc_cache : Optional[Dict[str, torch.Tensor]]
            Optional cache of encoder outputs (``memory``, ``mem_masks``,
            and ``precursors_all``) to avoid re-computation.

        Yields
        ------
        Dict[str, torch.Tensor]
            Batches of candidate database PSMs ready for scoring. Each batch
            contains repeated spectrum information for each candidate peptide
            to be scored against each spectrum.
        """
        device = self.decoder.device
        batch_size = batch["precursor_charge"].shape[0]

        # Iterate precursor charges and m/z values per spectrum.
        charge_iter = batch["precursor_charge"]  # tensor[B]
        mz_iter = batch["precursor_mz"]  # tensor[B]

        # Use pre-computed encoder outputs if available; otherwise compute once here.
        if enc_cache is None:
            mzs, ints, precursors_all, _ = self._process_batch(batch)
            memories, mem_masks = self.encoder(mzs, ints)
        else:
            memories, mem_masks = enc_cache["memory"], enc_cache["mem_masks"]
            precursors_all = enc_cache["precursors_all"]

        # Determine the candidates to score for each spectrum and
        # compile them into new batches with the same size as the original batch.
        candidates = []
        for i, (precursor_charge, precursor_mz) in enumerate(
            zip(charge_iter, mz_iter)
        ):
            for cand in self.protein_database.get_candidates(
                precursor_mz, precursor_charge
            ):
                candidates.append((i, cand))

            # Yield a batch if sufficient candidates are found or all spectra have been processed.
            while len(candidates) >= batch_size or (
                i == batch_size - 1 and len(candidates) > 0
            ):
                batch_candidates = candidates[:batch_size]

                # Repeat the spectrum information for each candidate to be matched.
                psm_batch = {key: [] for key in [*batch.keys(), "seq"]}
                for spec_i, cand in batch_candidates:
                    for key in batch.keys():
                        psm_batch[key].append(batch[key][spec_i])
                    psm_batch["seq"].append(cand)

                # Convert tensor items to batched tensors on the correct device.
                for key in psm_batch.keys():
                    if isinstance(psm_batch[key][0], torch.Tensor):
                        psm_batch[key] = torch.stack(psm_batch[key]).to(
                            self.decoder.device
                        )

                # Keep the original sequence string for downstream database lookup
                # (e.g., isoleucine ↔ leucine handling) and tokenize for scoring.
                psm_batch["original_seq_str"] = psm_batch["seq"]
                psm_batch["seq"] = self.tokenizer.tokenize(
                    psm_batch["seq"], add_stop=True
                ).to(self.decoder.device)

                # Attach the corresponding (pre)computed encoder outputs for these spectra.
                spec_idx = torch.tensor(
                    [i for i, _ in batch_candidates],
                    dtype=torch.long,
                    device=device,
                )
                psm_batch["memory"] = memories.index_select(0, spec_idx)
                psm_batch["mem_masks"] = mem_masks.index_select(0, spec_idx)
                psm_batch["precursors"] = precursors_all.index_select(
                    0, spec_idx
                )

                # Yield the PSM batch for processing.
                yield psm_batch

                # Remove the processed candidates and continue.
                candidates = candidates[batch_size:]


def _calc_match_score(
    batch_all_aa_scores: torch.Tensor,
    truth_aa_indices: torch.Tensor,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Calculate the score between the input spectra and associated
    peptide.

    This function now acts as a wrapper that prepares data for the unified
    _peptide_score function.

    Parameters
    ----------
    batch_all_aa_scores : torch.Tensor
        Amino acid scores for all amino acids in the vocabulary for
        every prediction made to generate the associated peptide (for an
        entire batch).
    truth_aa_indices : torch.Tensor
        Indices of the score for each actual amino acid in the peptide
        (for an entire batch).

    Returns
    -------
    peptide_scores: List[float]
        The peptide score for each PSM in the batch.
    aa_scores : List[np.ndarray]
        The amino acid scores for each PSM in the batch.
    """
    # Remove trailing token.
    batch_all_aa_scores = batch_all_aa_scores[:, :-1]

    # Get aa scores corresponding with true aas.
    per_aa_scores = torch.gather(
        batch_all_aa_scores, 2, truth_aa_indices.unsqueeze(-1)
    ).squeeze(-1)

    # Calculate peptide lengths.
    lengths = (truth_aa_indices != 0).sum(dim=1)

    # Fuse scores and lengths for a single GPU->CPU transfer.
    fused = torch.cat(
        [per_aa_scores, lengths.to(per_aa_scores.dtype).unsqueeze(1)], dim=1
    )
    fused_np = fused.detach().cpu().numpy()

    # Unpack scores and lengths on the CPU.
    per_aa_np = fused_np[:, :-1]
    lengths_np = fused_np[:, -1].astype(np.int32, copy=False)

    # Call the single, unified scoring function for batch calculation.
    # In database search mode, fits_precursor_mz is implicitly True.
    peptide_scores = _peptide_score(per_aa_np, lengths=lengths_np).tolist()

    # Extract AA scores for each peptide based on its length.
    B = per_aa_np.shape[0]
    aa_scores = [per_aa_np[i, : lengths_np[i]] for i in range(B)]

    return peptide_scores, aa_scores


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine
    shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the
    observed m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _peptide_score(
    aa_scores: np.ndarray,
    fits_precursor_mz: Union[bool, np.ndarray] = True,
    lengths: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate the peptide-level confidence score from the raw
    amino acid scores.

    The peptide score is the product of the raw amino acid scores.
    This function contains paths for both single peptide inputs
    (de novo mode) and batched peptide inputs (database search mode).

    Parameters
    ----------
    aa_scores : np.ndarray
        A 1D array of amino acid scores for a single peptide, or a 2D
        padded array for a batch of peptides.
    fits_precursor_mz : bool or np.ndarray
        Flag or array of flags indicating whether predictions fit the
        precursor m/z filter.
    lengths : Optional[np.ndarray]
        An array of peptide lengths, required when `aa_scores` is a 2D
        (batched) array.

    Returns
    -------
    peptide_score : float or np.ndarray
        The calculated peptide score or an array of scores for the batch.
    """
    eps = np.finfo(np.float64).eps

    # FAST PATH: de novo inference
    if aa_scores.ndim == 1:
        log_scores = np.log(np.clip(aa_scores, eps, 1))
        peptide_log_score = np.sum(log_scores)
        peptide_score = np.exp(peptide_log_score)

        if not fits_precursor_mz:
            peptide_score -= 1
        return peptide_score

    # BATCH PATH: database search
    else:
        if lengths is None:
            raise ValueError("`lengths` must be provided for batched input.")

        log_scores = np.log(np.clip(aa_scores, eps, 1))
        cumsum = np.cumsum(log_scores, axis=1)
        batch_size = aa_scores.shape[0]
        idx = np.arange(batch_size)
        peptide_log_scores = cumsum[idx, np.maximum(lengths - 1, 0)]
        peptide_scores = np.exp(peptide_log_scores)

        if isinstance(fits_precursor_mz, (bool, np.bool_)):
            if not fits_precursor_mz:
                peptide_scores -= 1
        else:
            peptide_scores[~fits_precursor_mz] -= 1

        return peptide_scores
