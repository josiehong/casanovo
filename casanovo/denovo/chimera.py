"""Chimera tokenizer and dataset for co-fragmented spectrum sequencing.

A chimeric annotation names the two peptides that were co-isolated into one
spectrum, written as ``PEP1:PEP2``. The model decodes them into two fixed
slots of decoder frames rather than into one string, so the separator never
enters the alphabet and never has to be predicted: it only ever appears in the
annotation file. That keeps the residue alphabet, and therefore the mass axis
the precise-mass-control search runs over, exactly what it is for
single-peptide sequencing.
"""

import os
from typing import Dict, Iterable, List, Tuple

import depthcharge.primitives
import depthcharge.tokenizers.peptides
import depthcharge.utils
import pandas as pd
import torch

from .dataloaders import AnnotatedSpectrumDataset

CHIMERIC_SEPARATOR = ":"


def _normalize_terminal_mods(
    peptide: depthcharge.primitives.Peptide,
) -> depthcharge.primitives.Peptide:
    """Replace empty terminal modification lists with None.

    Pyteomics' ``proforma.parse`` returns ``[]`` rather than ``None`` for an
    absent terminal modification in current versions. depthcharge's
    ``Peptide.split`` then emits a spurious ``[+0.000000]-`` / ``-[+0.000000]``
    token that is not in the alphabet.

    Parameters
    ----------
    peptide : depthcharge.primitives.Peptide
        The parsed peptide.

    Returns
    -------
    depthcharge.primitives.Peptide
        The same peptide, with absent terminal modifications set to None.
    """
    if peptide.modifications:
        if peptide.modifications[0] == []:
            peptide.modifications[0] = None
        if peptide.modifications[-1] == []:
            peptide.modifications[-1] = None
    return peptide


class ChimeraTokenizer(depthcharge.tokenizers.peptides.PeptideTokenizer):
    """A peptide tokenizer that understands chimeric annotations.

    The alphabet is unchanged from the parent. The only chimera-specific
    behaviour is ``split_annotation``, which divides ``"PEP1:PEP2"`` into its
    two peptides so that each can be tokenized on its own.

    Parameters
    ----------
    residues : Dict[str, float] | None
        Custom residue mass dictionary. If ``None``, uses the default
        canonical amino acids.
    replace_isoleucine_with_leucine : bool
        Replace isoleucine with leucine.
    reverse : bool
        Reverse peptide sequences during tokenization.
    start_token : str | None
        Start token string.
    stop_token : str | None
        Stop token string (default ``"$"``).
    chimeric_separator_token : str
        The character separating the two peptides of a chimeric annotation
        (default ``":"``). It is not a residue and is never tokenized.
    """

    def __init__(
        self,
        residues: Dict[str, float] | None = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: str | None = None,
        stop_token: str | None = "$",
        chimeric_separator_token: str = CHIMERIC_SEPARATOR,
    ) -> None:
        self.chimeric_separator_token = chimeric_separator_token

        super().__init__(
            residues=dict() if residues is None else residues,
            replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
            reverse=reverse,
            start_token=start_token,
            stop_token=stop_token,
        )

    # A class attribute rather than a closure installed on the instance.
    # Lightning pickles the tokenizer into the checkpoint hyperparameters,
    # a closure cannot be pickled, and Lightning drops what it cannot
    # pickle without failing. The tokenizer would then be missing from the
    # checkpoint and the alphabet equivalence check would quietly stop
    # running.
    @staticmethod
    def _parse_peptide(
        sequence: str,
    ) -> depthcharge.primitives.Peptide:
        """Parse a ProForma peptide, repairing empty terminal mod lists."""
        return _normalize_terminal_mods(
            depthcharge.primitives.Peptide.from_proforma(sequence)
        )

    def split_annotation(self, sequence: str) -> Tuple[str, str]:
        """Split a chimeric annotation into its two peptides.

        A separator is only recognized outside square brackets, since the
        same character names a controlled-vocabulary modification inside
        them, as in ``M[UNIMOD:35]``.

        Parameters
        ----------
        sequence : str
            An annotation, either ``"PEP1:PEP2"`` or a plain peptide.

        Returns
        -------
        Tuple[str, str]
            The two peptides. The second is the empty string when the
            annotation names only one.

        Raises
        ------
        ValueError
            If the annotation contains more than one separator.
        """
        parts, depth, start = [], 0, 0
        for i, char in enumerate(sequence):
            if char == "[":
                depth += 1
            elif char == "]":
                depth = max(0, depth - 1)
            elif char == self.chimeric_separator_token and depth == 0:
                parts.append(sequence[start:i])
                start = i + 1
        parts.append(sequence[start:])

        if len(parts) > 2:
            raise ValueError(
                f"Expected at most one "
                f"'{self.chimeric_separator_token}' in a chimeric "
                f"annotation, got {len(parts) - 1} in {sequence!r}"
            )
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]


class MskbChimeraTokenizer(ChimeraTokenizer):
    """A chimera tokenizer for MassIVE-KB annotations."""

    @staticmethod
    def _parse_peptide(
        sequence: str,
    ) -> depthcharge.primitives.Peptide:
        """Parse a MassIVE-KB peptide, repairing empty terminal mod lists."""
        return _normalize_terminal_mods(
            depthcharge.primitives.Peptide.from_massivekb(sequence)
        )


class ChimeraAnnotatedSpectrumDataset(AnnotatedSpectrumDataset):
    """An AnnotatedSpectrumDataset that tokenizes both chimeric peptides.

    The parent stores one tokenized target under ``batch["seq"]``. This
    subclass splits each annotation and stores the two peptides separately, in
    ``batch["seq"]`` and ``batch["seq_2"]``, which are the targets for the two
    decoder slots. Which peptide belongs in which slot is decided by the
    training objective, not here, so the order is simply the order written in
    the annotation.

    A spectrum annotated with a single peptide gets an all-padding second
    target, i.e. a target of length zero, which the CTC objective scores as
    the all-blank path. That is how the model learns to leave a slot empty.

    Parameters
    ----------
    spectra : pd.DataFrame | os.PathLike | Iterable[os.PathLike]
        Input spectra.
    annotations : str
        Name of the annotation column / field.
    tokenizer : ChimeraTokenizer
        A chimera-capable tokenizer.
    batch_size : int
        Batch size.
    path : os.PathLike, optional
        Optional path for the Lance index.
    parse_kwargs : Dict | None, optional
        Extra keyword arguments for the parser.
    **kwargs
        Additional arguments forwarded to the parent class.
    """

    def __init__(
        self,
        spectra: pd.DataFrame | os.PathLike | Iterable[os.PathLike],
        annotations: str,
        tokenizer: ChimeraTokenizer,
        batch_size: int,
        path: os.PathLike = None,
        parse_kwargs: Dict | None = None,
        **kwargs,
    ):
        super().__init__(
            spectra,
            annotations,
            tokenizer,
            batch_size,
            path,
            parse_kwargs,
            **kwargs,
        )

    def _tokenize_allowing_empty(self, sequences: List[str]) -> torch.Tensor:
        """Tokenize peptides, some of which may be the empty string.

        depthcharge cannot tokenize an empty sequence, so the empty ones are
        held out and their rows left as padding.

        Parameters
        ----------
        sequences : List[str]
            The peptides, possibly including empty strings.

        Returns
        -------
        torch.Tensor
            The tokenized peptides, padded to a common width.
        """
        filled = [i for i, seq in enumerate(sequences) if seq]
        if not filled:
            return torch.zeros((len(sequences), 1), dtype=torch.long)

        tokens = self.tokenizer.tokenize(
            [sequences[i] for i in filled],
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )
        if len(filled) == len(sequences):
            return tokens

        padded = torch.zeros(
            (len(sequences), tokens.shape[1]), dtype=tokens.dtype
        )
        padded[torch.tensor(filled)] = tokens
        return padded

    def _to_tensor(self, batch):
        """Convert a record batch to tensors, splitting chimeric annotations.

        Overrides the parent to store the two peptides of each annotation as
        two separate targets rather than one tokenized string.
        """
        batch = super(AnnotatedSpectrumDataset, self)._to_tensor(batch)

        first, second = [], []
        for sequence in batch[self.annotations]:
            one, two = self.tokenizer.split_annotation(sequence)
            first.append(one)
            second.append(two)

        batch[self.annotations] = self._tokenize_allowing_empty(first)
        batch[self.annotations + "_2"] = self._tokenize_allowing_empty(second)
        return batch
