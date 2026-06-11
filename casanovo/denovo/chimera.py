"""Tokenizer and dataset classes supporting chimeric spectra.

A chimeric spectrum contains two co-fragmenting peptides. Their sequences are
encoded as a single string joined by a separator token (``:`` by default),
e.g. ``PEPTIDEA:PEPTIDEB``. Because the ordering of the two peptides is
arbitrary, the dataset emits both the sequence and its *complement* (the two
peptides in swapped order) so that a permutation-invariant loss can be computed
during training (see :mod:`casanovo.denovo.model`).
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import depthcharge.constants
import depthcharge.primitives
import depthcharge.utils
import pandas as pd
import torch
from depthcharge.data import AnnotatedSpectrumDataset
from depthcharge.tokenizers.peptides import PeptideTokenizer


class ChimeraTokenizer(PeptideTokenizer):
    """A peptide tokenizer that understands a chimeric separator token.

    Parameters
    ----------
    residues : Dict[str, float] | None
        Additional residues to add to the vocabulary, mapping the residue to
        its monoisotopic mass.
    replace_isoleucine_with_leucine : bool
        Replace I with L residues, since they are isobaric.
    reverse : bool
        Reverse the sequence for tokenization.
    start_token : str | None
        The start token to use.
    stop_token : str | None
        The stop token to use.
    chimeric_separator_token : str
        The token used to separate the two peptides of a chimeric annotation.
    """

    def __init__(
        self,
        residues: Optional[Dict[str, float]] = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: Optional[str] = None,
        stop_token: Optional[str] = "$",
        chimeric_separator_token: str = ":",
    ) -> None:
        self.chimeric_separator_token = chimeric_separator_token
        residues = dict() if residues is None else dict(residues)
        residues[chimeric_separator_token] = 0.0

        super().__init__(
            residues=residues,
            replace_isoleucine_with_leucine=replace_isoleucine_with_leucine,
            reverse=reverse,
            start_token=start_token,
            stop_token=stop_token,
        )

    def _split_on_separator(self, sequence: str) -> list[str]:
        """Split a sequence on the chimeric separator token.

        Only a separator occurring at the top level (outside any ``[...]``
        modification group) is treated as a peptide separator. This avoids
        splitting inside ProForma modifications whose notation may reuse the
        separator character (e.g. a controlled-vocabulary accession such as
        ``[UNIMOD:35]`` when the separator is ``:``).
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
        return parts

    def compliment(
        self,
        sequences: Iterable[str] | str,
    ) -> Iterable[str]:
        """Get complement sequences (the two peptides in swapped order)."""
        compliment_sequences = []
        for seq in depthcharge.utils.listify(sequences):
            peptides = self._split_on_separator(seq)
            compliment = self.chimeric_separator_token.join(peptides[::-1])
            compliment_sequences.append(compliment)

        return compliment_sequences

    def tokenize_compliment(
        self,
        sequences: Iterable[str] | str,
        add_start: bool = False,
        add_stop: bool = False,
        to_strings: bool = False,
    ) -> torch.tensor | List[List[str]]:
        """Tokenize complement sequences."""
        return self.tokenize(
            self.compliment(sequences),
            add_start=add_start,
            add_stop=add_stop,
            to_strings=to_strings,
        )

    def split(self, sequence: str) -> list[str]:
        """Split a (possibly chimeric) peptide sequence into tokens."""
        peptides = self._split_on_separator(sequence)
        if len(peptides) in [1, 2]:
            split = super().split(peptides[0])
            if len(peptides) == 2:
                split += [self.chimeric_separator_token]
                split += super().split(peptides[1])
        else:
            raise ValueError(
                f"Sequence {sequence} contains more than one chimeric "
                "separator; sequences can contain at most one separator."
            )

        return split

    def calculate_precursor_ions(
        self,
        tokens: torch.Tensor | Iterable[str],
        charges: torch.Tensor,
        charges_two: Optional[torch.Tensor] = None,
        give_max_mz: bool = False,
    ) -> torch.Tensor:
        """Calculate the m/z for precursor ions.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, len_seq)
            The tokens corresponding to the peptide sequence.
        charges : torch.Tensor of shape (n_sequences,)
            The charge state for each (first) peptide.
        charges_two : torch.Tensor of shape (n_sequences,), optional
            The charge state for the second peptide of each chimera. Required
            when ``give_max_mz`` is ``True``.
        give_max_mz : bool
            Whether to return the max m/z over the two peptides in a chimera
            (``True``) or just the m/z of the first peptide (``False``).

        Returns
        -------
        torch.Tensor of shape (n_sequences,)
            The monoisotopic m/z for each charged peptide.
        """
        if isinstance(tokens[0], str):
            tokens = self.tokenize(depthcharge.utils.listify(tokens))

        if not isinstance(charges, torch.Tensor):
            charges = torch.tensor(charges)
            if not charges.shape:
                charges = charges[None]

        chimera_separator = self.index[self.chimeric_separator_token]
        masses = self.masses[tokens].cumsum(dim=1)
        is_separator = tokens == chimera_separator
        is_chimeric = is_separator.sum(dim=1)
        if is_chimeric.max().item() > 1:
            raise ValueError(
                "Sequences can contain at most one chimeric separator."
            )

        is_chimeric = is_chimeric.to(torch.bool)
        mass_one = (masses * is_separator).sum(dim=1, keepdim=True)
        mass_one[~is_chimeric] = masses[~is_chimeric, -1].unsqueeze(1)
        mz_one = (mass_one + depthcharge.constants.H2O) / charges
        mz_one += depthcharge.constants.PROTON

        if give_max_mz:
            if charges_two is None:
                raise ValueError(
                    "charges_two must be given if using give_max_mz"
                )

            mass_two = masses[:, -1] - mass_one
            mz_two = mass_two[is_chimeric] + depthcharge.constants.H2O
            mz_two /= charges_two[is_chimeric]
            mz_two += depthcharge.constants.PROTON

            calc_mz = torch.cat((mz_one, mz_two), dim=1)
            calc_mz = calc_mz.max(dim=1).values
        else:
            calc_mz = mz_one.squeeze(-1)

        return calc_mz


class MskbChimeraTokenizer(ChimeraTokenizer):
    """A chimeric tokenizer that parses MassIVE-KB peptide annotations."""

    _parse_peptide = depthcharge.primitives.Peptide.from_massivekb


class ChimeraAnnotatedSpectrumDataset(AnnotatedSpectrumDataset):
    """An annotated spectrum dataset that emits complement sequences.

    See :class:`depthcharge.data.AnnotatedSpectrumDataset`. In addition to the
    tokenized ``seq``, each batch also contains ``seq_compliment`` (the
    tokenized complement sequence) and ``precursor_charge_two`` (the charge of
    the second peptide of each chimera).
    """

    def __init__(
        self,
        spectra: pd.DataFrame | os.PathLike | Iterable[os.PathLike],
        annotations: str,
        tokenizer: ChimeraTokenizer,
        batch_size: int,
        path: os.PathLike = None,
        parse_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            spectra=spectra,
            annotations=annotations,
            tokenizer=tokenizer,
            batch_size=batch_size,
            path=path,
            parse_kwargs=parse_kwargs,
            **kwargs,
        )

    def _to_tensor(self, batch):
        """Convert a record batch to tensors.

        See :meth:`depthcharge.data.AnnotatedSpectrumDataset._to_tensor`.
        """
        # Skip AnnotatedSpectrumDataset._to_tensor (which would tokenize the
        # annotation with the standard logic) and tokenize seq, its complement,
        # and the second charge ourselves.
        batch = super(AnnotatedSpectrumDataset, self)._to_tensor(batch)
        sequence = batch[self.annotations]
        batch[self.annotations] = self.tokenizer.tokenize(
            sequence,
            add_start=self.tokenizer.start_token is not None,
            add_stop=self.tokenizer.stop_token is not None,
        )
        batch[self.annotations + "_compliment"] = (
            self.tokenizer.tokenize_compliment(
                sequence,
                add_start=self.tokenizer.start_token is not None,
                add_stop=self.tokenizer.stop_token is not None,
            )
        )
        batch["precursor_charge_two"] = torch.tensor(
            [int(charge[0]) for charge in batch["charge_two"]], dtype=int
        )

        return batch
