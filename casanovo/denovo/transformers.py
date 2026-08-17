"""Transformer encoder and decoder for the de novo sequencing task."""

from collections.abc import Callable

import torch
from depthcharge.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.tokenizers import Tokenizer
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    SpectrumTransformerEncoder,
)


class PeptideDecoder(AnalyteTransformerDecoder):
    """
    A transformer decoder for peptide sequences.

    Parameters
    ----------
    n_tokens : int
        The number of tokens used to tokenize peptide sequences.
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass
        spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the
        Transformer layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    positional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False``
        disables positional encodings, typically only for ablation
        tests.
    padding_int : int or None, optional
        The index that represents padding in the input sequence.
        Required only if ``n_tokens`` was provided as an ``int``.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        padding_int: int | None = None,
        max_charge: int = 4,
    ) -> None:
        """Initialize a PeptideDecoder."""

        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
            padding_int=padding_int,
        )

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mass_encoder = FloatEncoder(d_model)

        # Override the output layer to have +1 in the second dimension
        # compared to the AnalyteTransformerDecoder to account for
        # padding as a possible class (=0) and avoid problems during
        # beam search decoding.
        self.final = torch.nn.Linear(
            d_model, self.token_encoder.num_embeddings
        )

    def global_token_hook(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Override global_token_hook to include precursor information.

        Parameters
        ----------
        *args :
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the
            next token. Optionally, these may be the token indices
            instead of a string.
        precursors : torch.Tensor
            Precursor information.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The global token representations.
        """
        masses = self.mass_encoder(precursors[:, None, 0]).squeeze(1)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges
        return precursors


class SpectrumEncoder(SpectrumTransformerEncoder):
    """
    A Transformer encoder for input mass spectra.

    Parameters
    ----------
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass
        spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the
        Transformer layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : PeakEncoder or bool, optional
        The function to encode the (m/z, intensity) tuples of each mass
        spectrum. `True` uses the default sinusoidal encoding and `False`
        instead performs a 1 to `d_model` learned linear projection.
    claim_channel : bool, optional
        Add a per-peak claim-flag input channel: a learned vector added to
        the embedding of every peak flagged as claimed (i.e. matched by a
        previously decoded peptide's fragment ions). Zero-initialized, so
        at initialization the encoder is identical to one without the
        channel and a claim-less checkpoint can be warm-started.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
        claim_channel: bool = False,
    ):
        """Initialize a SpectrumEncoder."""
        super().__init__(
            d_model, n_head, dim_feedforward, n_layers, dropout, peak_encoder
        )

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))
        if claim_channel:
            self.claim_vec = torch.nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        claims: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embed a batch of mass spectra, optionally with peak-claim flags.

        Identical to ``SpectrumTransformerEncoder.forward`` except that
        ``claims`` (bool tensor of shape (n_spectra, n_peaks), or None)
        adds ``claim_vec`` to the embedding of every flagged peak.
        """
        spectra = torch.stack([mz_array, intensity_array], dim=2)

        src_key_padding_mask = spectra.sum(dim=2) == 0
        global_token_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(
            src_key_padding_mask
        )
        src_key_padding_mask = torch.cat(
            [global_token_mask, src_key_padding_mask], dim=1
        )

        peaks = self.peak_encoder(spectra)
        if claims is not None:
            peaks = peaks + claims.to(peaks.dtype)[:, :, None] * self.claim_vec

        latent_spectra = self.global_token_hook(
            *args,
            mz_array=mz_array,
            intensity_array=intensity_array,
            **kwargs,
        )

        peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)
        out = self.transformer_encoder(
            peaks,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return out, src_key_padding_mask

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Override global_token_hook to include latent_spectrum parameter.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, max_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, max_peaks)
            The zero-padded intensity dimension for a batch of mass
            spectra.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.

        """
        return self.latent_spectrum.squeeze(0).expand(mz_array.shape[0], -1)
