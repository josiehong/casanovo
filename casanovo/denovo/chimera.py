"""Tokenizer and dataset classes supporting chimeric spectra.

A chimeric spectrum contains two co-fragmenting peptides. Their sequences are
encoded as a single string joined by a separator token (``+`` by default, the
ProForma chimeric joiner), e.g. ``PEPTIDEA+PEPTIDEB``. Because the ordering of
the two peptides is arbitrary, the dataset emits both the sequence and its
*complement* (the two peptides in swapped order) so that a permutation-invariant
loss can be computed during training (see :mod:`casanovo.denovo.model`).
"""

from typing import Dict, Iterable, List, Optional

import depthcharge.primitives
import depthcharge.utils
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
        The token joining the two peptides in the raw annotation string
        (``PEPTIDEA+PEPTIDEB``). It is *not* part of the token vocabulary. This
        branch reuses the stop token as the peptide boundary, so a chimera
        tokenizes to ``pep1 <stop> pep2 <stop>`` and the vocabulary matches a
        standard (non-chimeric) tokenizer.
    """

    def __init__(
        self,
        residues: Optional[Dict[str, float]] = None,
        replace_isoleucine_with_leucine: bool = False,
        reverse: bool = False,
        start_token: Optional[str] = None,
        stop_token: Optional[str] = "$",
        chimeric_separator_token: str = "+",
    ) -> None:
        self.chimeric_separator_token = chimeric_separator_token
        residues = dict() if residues is None else dict(residues)

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
        separator character (e.g. a mass modification such as ``[+15.99]`` when
        the separator is the ProForma chimeric joiner ``+``).

        We split here because the tokenizer's ProForma backend cannot tokenize a
        string containing a top-level ``+`` (see ``split`` below). pyteomics
        added opt-in chimeric parsing (``proforma.parse(seq, chimeric=True)``),
        but Depthcharge does not pass ``chimeric=True``, so the top-level split
        is still performed here and each peptide is tokenized independently.

        TODO(depthcharge-chimeric): Once Depthcharge tokenizes chimeric ProForma
        natively, this helper is only needed for ``compliment`` (which swaps the
        component order in the raw string); ``split`` can drop it entirely.
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
        """Split a (possibly chimeric) peptide sequence into tokens.

        A chimeric sequence is split on the top-level separator and each
        peptide is tokenized independently via the parent (ProForma) tokenizer,
        with the separator token re-inserted between them. We cannot hand the
        whole ``A+B`` string to the parent because Depthcharge's
        ``PeptideTokenizer.split`` routes through
        ``Peptide.from_proforma -> proforma.parse(seq)`` *without*
        ``chimeric=True``, so the ProForma backend rejects the top-level ``+``.

        TODO(depthcharge-chimeric): When Depthcharge is updated to support
        chimeric ProForma, this whole override can be removed and the parent
        ``PeptideTokenizer.split`` used directly. Depthcharge needs to:
          1. Call ``proforma.parse(seq, chimeric=True)`` /
             ``Peptide.from_proforma(seq, chimeric=True)`` (pyteomics with
             opt-in chimeric parsing), which returns one parse result per
             component instead of raising on the top-level ``+``.
          2. Tokenize each returned component and join them with the chimeric
             separator token (kept in the vocabulary by ``__init__``).
        Until then, keep this manual split. Note ``compliment`` still needs
        ``_split_on_separator`` to swap the component order in the raw string.
        """
        peptides = self._split_on_separator(sequence)
        if len(peptides) in [1, 2]:
            split = super().split(peptides[0])
            if len(peptides) == 2:
                # The stop token is the boundary; ``tokenize`` appends the
                # terminating stop, giving ``pep1 <stop> pep2 <stop>``.
                split += [self.stop_token]
                split += super().split(peptides[1])
        else:
            raise ValueError(
                f"Sequence {sequence} contains more than one chimeric "
                "separator; sequences can contain at most one separator."
            )

        return split

    # NOTE: ``calculate_precursor_ions`` is intentionally not overridden. It is
    # only ever called on already-split single peptides (see
    # ``Spec2Pep.on_predict_batch_end``), for which the inherited
    # ``PeptideTokenizer`` implementation is correct. A chimera-aware version
    # (splitting precursor mass across the two peptides) was dropped as unused.


class MskbChimeraTokenizer(ChimeraTokenizer):
    """A chimeric tokenizer that parses MassIVE-KB peptide annotations."""

    _parse_peptide = depthcharge.primitives.Peptide.from_massivekb


class ChimeraAnnotatedSpectrumDataset(AnnotatedSpectrumDataset):
    """An annotated spectrum dataset that emits complement sequences.

    See :class:`depthcharge.data.AnnotatedSpectrumDataset`. In addition to the
    tokenized ``seq``, each batch also contains ``seq_compliment`` (the
    tokenized complement sequence) used for the permutation-invariant loss.
    """

    def _to_tensor(self, batch):
        """Convert a record batch to tensors.

        See :meth:`depthcharge.data.AnnotatedSpectrumDataset._to_tensor`.
        """
        # Skip AnnotatedSpectrumDataset._to_tensor (which would tokenize the
        # annotation with the standard logic) and tokenize seq and its
        # complement ourselves.
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

        return batch
