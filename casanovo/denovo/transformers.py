"""Transformer encoder and decoder for the de novo sequencing task."""

from collections.abc import Callable, Sequence

import torch
from depthcharge.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.tokenizers import Tokenizer
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    SpectrumTransformerEncoder,
)


def _peak_summary(
    memory: torch.Tensor, memory_key_padding_mask: torch.Tensor | None
) -> torch.Tensor:
    """
    The mean embedding of each spectrum's real peaks.

    Row 0 of the encoder output is its learned global token and is left
    out; padded peaks are excluded, and a mask marks padding with True.
    """
    peaks = memory[:, 1:]
    if memory_key_padding_mask is None:
        return peaks.mean(dim=1)
    keep = (~memory_key_padding_mask[:, 1:]).unsqueeze(-1).to(peaks.dtype)
    return (peaks * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)


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
        self_cond_layers: Sequence[int] = (),
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

        # Override the output layer with one class beyond the token
        # embeddings (which include padding at index 0): the last index
        # serves as the dedicated CTC blank class.
        self.final = torch.nn.Linear(
            d_model, self.token_encoder.num_embeddings + 1
        )

        # Self-conditioning: the layers after which this decoder scores its
        # own hidden states and feeds the prediction back in. Empty means the
        # decoder behaves exactly as it did before and grows no parameters,
        # so a checkpoint trained without it still loads.
        self.self_cond_layers = tuple(
            k for k in sorted(set(self_cond_layers)) if 1 <= k < n_layers
        )
        if self.self_cond_layers:
            # Maps a distribution over the vocabulary back to model space.
            # No bias: a constant offset would be the same at every frame
            # and could be absorbed by the layer that follows.
            self.cond_proj = torch.nn.Linear(
                self.final.out_features, d_model, bias=False
            )
        else:
            self.cond_proj = None

        # FiLM on the positional encoding (Perez et al., AAAI 2018): each
        # frame's input is PE(t), since its token embedding is zero, and
        # this rescales and shifts it per channel by amounts computed from
        # the mean peak embedding. PE is the only thing that tells frames
        # apart, so the scale is 1 + tanh(.), which stays positive and
        # cannot erase it. Zero-init makes the scale 1 and the shift 0, so
        # an untrained decoder is fed exactly the positional encoding, as
        # the decoder without FiLM is.
        self.pe_film = torch.nn.Linear(d_model, 2 * d_model)
        torch.nn.init.zeros_(self.pe_film.weight)
        torch.nn.init.zeros_(self.pe_film.bias)

    def _decoder_input(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        The decoder input: the precursor token, then one row per frame.

        ``embed`` and ``forward_self_conditioned`` both build their input
        here so the two decode paths cannot drift apart.
        """
        encoded = self.token_encoder(tokens)
        global_token = self.global_token_hook(tokens, *args, **kwargs)
        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)
        encoded = self.positional_encoder(encoded)

        summary = _peak_summary(memory, memory_key_padding_mask)
        scale, shift = self.pe_film(summary).chunk(2, dim=-1)
        frames = (1 + torch.tanh(scale))[:, None, :] * encoded[:, 1:]
        frames = frames + shift[:, None, :]
        return torch.cat([encoded[:, :1], frames], dim=1)

    def forward_self_conditioned(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Decode, scoring the stack's own hidden states along the way.

        Self-conditioned CTC (Nozaki and Komatsu, Interspeech 2021):
        after each layer in ``self_cond_layers`` the hidden states are
        scored with the same output layer the model already has, and that
        prediction is projected back to model space and added to the
        states before the next layer runs. Positions are therefore no
        longer predicted independently of one another, at the cost of one
        auxiliary loss per conditioning layer during training and nothing
        at inference.

        This has to open up the layer stack rather than call
        ``transformer_decoder`` in one shot, so it repeats the mask
        preparation from ``embed``: the all-False target mask that makes
        attention non-causal, and no key padding mask, since every frame
        is a real decoding slot. See ``embed`` for why the superclass's
        inferred padding mask is wrong here.

        Returns
        -------
        scores : torch.Tensor of shape (batch, len_seq, n_tokens)
            The final-layer scores, identical in meaning to ``forward``.
        intermediates : list of torch.Tensor
            One score tensor per conditioning layer, for the auxiliary
            CTC losses. Empty when self-conditioning is off, in which
            case the scores match ``forward`` exactly.
        """
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        encoded = self._decoder_input(
            tokens, memory, memory_key_padding_mask, *args, **kwargs
        )

        # Non-causal attention, as in `embed` above: every frame sees
        # every other frame. No key padding mask for the same reason
        # given there.
        length = encoded.shape[1]
        tgt_mask = torch.zeros(
            (length, length), dtype=torch.bool, device=encoded.device
        )

        intermediates = []
        for depth, layer in enumerate(self.transformer_decoder.layers, 1):
            encoded = layer(
                encoded,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if depth in self.self_cond_layers:
                scores = self.final(encoded)
                intermediates.append(scores)
                encoded = encoded + self.cond_proj(scores.softmax(dim=-1))

        if self.transformer_decoder.norm is not None:
            encoded = self.transformer_decoder.norm(encoded)
        return self.final(encoded), intermediates

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

    def embed(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Embed the decoder frames with full, non-causal attention.

        ``tgt_mask`` is all False, so every frame sees every other frame.
        ``tgt_key_padding_mask`` is None, which is why this reimplements
        the superclass rather than delegating to it: the superclass infers
        padding as ``encoded.sum(axis=2) == 0``, and every frame here is
        fed token id 0, ``padding_idx``, whose embedding is zero. That
        marked every frame as padding, so no frame could attend to
        another, when none of them is padding.
        """
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        encoded = self._decoder_input(
            tokens, memory, memory_key_padding_mask, *args, **kwargs
        )

        if tgt_mask is None:
            length = encoded.shape[1]
            tgt_mask = torch.zeros(
                (length, length), dtype=torch.bool, device=encoded.device
            )

        return self.transformer_decoder(
            tgt=encoded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
        )


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
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
    ):
        """Initialize a SpectrumEncoder."""
        super().__init__(
            d_model, n_head, dim_feedforward, n_layers, dropout, peak_encoder
        )

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

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
