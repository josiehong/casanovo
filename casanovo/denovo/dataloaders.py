"""Data loaders for the de novo sequencing task."""

import functools
import logging
import os
import pathlib
from typing import Callable, Optional, Sequence

import lance
import lightning.pytorch as pl
import numpy as np
import pyarrow as pa
import spectrum_utils.spectrum as sus
import torch.utils.data._utils.collate
from depthcharge.data import (
    AnnotatedSpectrumDataset,
    CustomField,
    SpectrumDataset,
    preprocessing,
)
from depthcharge.tokenizers import PeptideTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe


logger = logging.getLogger("casanovo")


class DeNovoDataModule(pl.LightningDataModule):
    """
    Data loader to prepare MS/MS spectra for a Spec2Pep predictor.

    Parameters
    ----------
    lance_dir : str
        Directory to store Lance spectrum index files.
    train_paths : Sequence[str], optional
        Spectrum Lance path(s) for model training.
    valid_paths : Sequence[str], optional
        Spectrum Lance path(s) for validation.
    test_paths : Sequence[str], optional
        Spectrum Lance path(s) for evaluation or inference.
    train_batch_size : int
        The batch size to use for training.
    eval_batch_size : int
        The batch size to use for inference.
    min_peaks : Optional[int]
        The number of peaks for a spectrum to be considered valid.
    max_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum.
        `None` retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to
        exclude TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage
        of the base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around
        the precursor mass.
    max_charge: int
        Remove PSMs which precursor charge higher than specified
        max_charge.
    tokenizer: Optional[PeptideTokenizer]
        Tokenizer for processing peptide sequences.
    shuffle: Optional[bool]
        Shuffle the training dataset or not. Default is True.
    shuffle_buffer_size: Optional[int]
        Number of samples to buffer for randomly shuffling the training
        data.
    n_workers : int, optional
        The number of workers to use for data loading. By default, the
        number of available CPU cores on the current machine is used.
    chimera_curriculum : int, optional
        If set, train with a chimeric curriculum (see
        ``CurriculumDataset``) holding at most this many chimeric spectra
        per batch.
    epoch_fn : Callable[[], int], optional
        Returns the current 0-based training epoch, for the curriculum.
    """

    def __init__(
        self,
        lance_dir: str,
        train_paths: Optional[Sequence[str]] = None,
        valid_paths: Optional[Sequence[str]] = None,
        test_paths: Optional[Sequence[str]] = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 1028,
        min_peaks: Optional[int] = 20,
        max_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        max_charge: Optional[int] = 10,
        tokenizer: Optional[PeptideTokenizer] = None,
        shuffle: Optional[bool] = True,
        shuffle_buffer_size: Optional[int] = 10_000,
        n_workers: Optional[int] = None,
        chimera_curriculum: Optional[int] = None,
        epoch_fn: Optional[Callable[[], int]] = None,
    ):
        super().__init__()

        self.lance_dir = lance_dir
        self.chimera_curriculum = chimera_curriculum
        self.epoch_fn = epoch_fn

        self.train_paths = train_paths
        self.valid_paths = valid_paths
        self.test_paths = test_paths

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # Spectrum preprocessing functions.
        self.preprocessing_fn = [
            preprocessing.set_mz_range(min_mz=min_mz, max_mz=max_mz),
            preprocessing.remove_precursor_peak(remove_precursor_tol, "Da"),
            preprocessing.scale_intensity("root", 1),
            preprocessing.filter_intensity(min_intensity, max_peaks),
            functools.partial(_discard_low_quality, min_peaks=min_peaks),
            _scale_to_unit_norm,
        ]
        self.valid_charge = np.arange(1, max_charge + 1)

        self.tokenizer = tokenizer or PeptideTokenizer()

        # Set to None to disable shuffling, otherwise Torch throws an error.
        self.shuffle = shuffle if shuffle else None
        self.shuffle_buffer_size = shuffle_buffer_size

        self.n_workers = n_workers if n_workers is not None else os.cpu_count()

        # Custom fields to read from the input files.
        self.custom_field_anno = CustomField(
            "seq", lambda x: x["params"]["seq"], pa.string()
        )

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.protein_database = None

    def setup(self, stage: str = None, annotated: bool = True) -> None:
        """
        Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are
            prepared by default.
        annotated: bool
            True if peptide sequence annotations are available for the
            test data.
        """
        if stage in (None, "fit", "validate"):
            if self.train_paths is not None and self.chimera_curriculum:
                self.train_dataset = self._make_curriculum_dataset()
            elif self.train_paths is not None:
                self.train_dataset = self._make_dataset(
                    self.train_paths,
                    annotated=True,
                    mode="train",
                    shuffle=self.shuffle,
                )
            if self.valid_paths is not None:
                self.valid_dataset = self._make_dataset(
                    self.valid_paths,
                    annotated=True,
                    mode="valid",
                    shuffle=False,
                )
        if stage in (None, "test"):
            if self.test_paths is not None:
                self.test_dataset = self._make_dataset(
                    self.test_paths,
                    annotated=annotated,
                    mode="test",
                    shuffle=False,
                )

    def _make_dataset(
        self, paths, annotated, mode, shuffle
    ) -> torch.utils.data.Dataset:
        """
        Make spectrum datasets.

        Parameters
        ----------
        paths : Iterable[str]
            Paths to read the spectrum input data from.
        annotated: bool
            True if peptide sequence annotations are available for the
            test data.
        mode: str {"train", "valid", "test"}
            The mode indicating name of lance instance
        shuffle: bool
            Shuffle the dataset or not.

        Returns
        -------
        torch.utils.data.Dataset
            A PyTorch Dataset for the given peak files.
        """
        custom_fields = [self.custom_field_anno] if annotated else []
        lance_path = pathlib.Path(f"{self.lance_dir}/{mode}.lance")

        parse_params = dict(
            preprocessing_fn=self.preprocessing_fn,
            valid_charge=self.valid_charge,
            custom_fields=custom_fields,
        )

        dataset_params = dict(
            batch_size=(
                self.train_batch_size
                if mode == "train"
                else self.eval_batch_size
            )
        )
        anno_dataset_params = dataset_params | dict(
            tokenizer=self.tokenizer,
            annotations="seq",
        )

        # Imported here rather than at module scope because `chimera` imports
        # `AnnotatedSpectrumDataset` from this module.
        from .chimera import ChimeraAnnotatedSpectrumDataset, ChimeraTokenizer

        if annotated:
            params = anno_dataset_params
            if isinstance(self.tokenizer, ChimeraTokenizer):
                Dataset = ChimeraAnnotatedSpectrumDataset
            else:
                Dataset = AnnotatedSpectrumDataset
        else:
            Dataset, params = SpectrumDataset, dataset_params

        if (
            len(paths) == 1
            and pathlib.Path(paths[0]).suffix.lower() == ".lance"
        ):
            dataset = Dataset.from_lance(paths[0], **params)
        else:
            dataset = Dataset(
                spectra=paths,
                path=lance_path,
                parse_kwargs=parse_params,
                **params,
            )

        if shuffle:
            dataset = ShufflerIterDataPipe(
                dataset, buffer_size=self.shuffle_buffer_size
            )

        return dataset

    def _make_curriculum_dataset(self) -> "CurriculumDataset":
        """
        Make the training dataset for the chimeric curriculum.

        The training spectra are parsed once, then read back as two
        subsets, single-peptide and chimeric, by a filter on the
        annotation separator.

        Returns
        -------
        CurriculumDataset
            The training dataset.
        """
        from .chimera import ChimeraAnnotatedSpectrumDataset

        path = str(
            self._make_dataset(
                self.train_paths, annotated=True, mode="train", shuffle=False
            ).path
        )
        separator = self.tokenizer.chimeric_separator_token
        chimeric = f"seq LIKE '%{separator}%'"
        # The filter also matches a separator inside a modification such as
        # "[UNIMOD:35]", so check it against the tokenizer's own split.
        seqs = lance.dataset(path).to_table(columns=["seq"])["seq"]
        n_chimeric = sum(
            bool(self.tokenizer.split_annotation(seq)[1])
            for seq in seqs.to_pylist()
        )
        if lance.dataset(path).count_rows(filter=chimeric) != n_chimeric:
            raise ValueError(
                f"Some annotations contain '{separator}' inside a "
                "modification, so the chimeric curriculum cannot tell "
                "chimeric spectra apart by filter."
            )

        def make_subset(is_chimeric, batch_size):
            subset = ChimeraAnnotatedSpectrumDataset.from_lance(
                path,
                annotations="seq",
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                filter=chimeric if is_chimeric else f"NOT ({chimeric})",
            )
            if self.shuffle:
                subset = ShufflerIterDataPipe(
                    subset, buffer_size=self.shuffle_buffer_size
                )
            return subset

        return CurriculumDataset(
            make_subset,
            self.train_batch_size,
            self.chimera_curriculum,
            self.epoch_fn,
        )

    def _make_loader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        shuffle : bool
            Option to shuffle the batches.

        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=self.n_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        # The curriculum shuffles inside its subsets.
        shuffle = None if self.chimera_curriculum else self.shuffle
        return self._make_loader(self.train_dataset, shuffle=shuffle)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset)

    def db_dataloader(self) -> torch.utils.data.DataLoader:
        """Get a special dataloader for DB search."""
        return self._make_loader(self.test_dataset)


class CurriculumDataset(torch.utils.data.IterableDataset):
    """
    Training batches whose chimeric share grows by one spectrum per epoch.

    Epoch ``e`` (0-based) puts ``min(e + 1, max_chimeric)`` chimeric
    spectra in every batch and fills the rest with single-peptide ones.
    An epoch is one pass over the single-peptide spectra; the chimeric
    ones are cycled.

    Parameters
    ----------
    make_subset : Callable[[bool, int], Iterable[dict]]
        Builds the chimeric (True) or single-peptide (False) spectra, in
        batches of the given size.
    batch_size : int
        The number of spectra per batch.
    max_chimeric : int
        The most chimeric spectra per batch.
    epoch_fn : Callable[[], int], optional
        Returns the current 0-based epoch. By default, the passes over
        this dataset are counted.
    """

    def __init__(
        self,
        make_subset: Callable,
        batch_size: int,
        max_chimeric: int,
        epoch_fn: Optional[Callable[[], int]] = None,
    ):
        super().__init__()
        self.make_subset = make_subset
        self.batch_size = batch_size
        self.max_chimeric = max_chimeric
        self.epoch_fn = epoch_fn
        self._passes = 0

    def __iter__(self):
        epoch = self.epoch_fn() if self.epoch_fn is not None else self._passes
        self._passes += 1
        n_chimeric = min(epoch + 1, self.max_chimeric, self.batch_size - 1)
        logger.info(
            "Epoch %d: %d of %d spectra per batch are chimeric",
            epoch,
            n_chimeric,
            self.batch_size,
        )
        chimeric = self._cycle_chimeric(n_chimeric)
        singles = self.make_subset(False, self.batch_size - n_chimeric)
        for batch in singles:
            yield _concat_batches(batch, next(chimeric))

    def _cycle_chimeric(self, batch_size: int):
        """Yield full batches of chimeric spectra, pass after pass."""
        while True:
            full = False
            for batch in self.make_subset(True, batch_size):
                # A pass ends in a short batch; skip it to keep the count.
                if len(batch["seq"]) == batch_size:
                    full = True
                    yield batch
            if not full:
                raise ValueError(
                    f"Fewer than {batch_size} chimeric training spectra."
                )


def _concat_batches(first: dict, second: dict) -> dict:
    """
    Join two batches, padding 2-D tensors to a common width.

    Parameters
    ----------
    first, second : dict
        Batches with the same keys.

    Returns
    -------
    dict
        The spectra of ``first`` followed by those of ``second``.
    """
    batch = {}
    for key, value in first.items():
        other = second[key]
        if not isinstance(value, torch.Tensor):
            batch[key] = value + other
            continue
        if value.dim() > 1:
            width = max(value.shape[1], other.shape[1])
            value, other = (
                torch.nn.functional.pad(t, (0, width - t.shape[1]))
                for t in (value, other)
            )
        batch[key] = torch.cat([value, other])
    return batch


def _discard_low_quality(
    spectrum: sus.MsmsSpectrum, min_peaks: int
) -> sus.MsmsSpectrum:
    """
    Discard low quality spectra.

    Spectra are considered low quality if:
    - They have fewer than 20 peaks.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum to check for low quality.
    min_peaks : int
        The minimum number of peaks required for a spectrum to be
        considered high quality.

    Returns
    -------
    sus.MsmsSpectrum
        The spectrum if it is of high quality, otherwise None.

    Raises
    ------
    ValueError
        If the spectrum is of low quality.
    """
    if len(spectrum.mz) < min_peaks:
        raise ValueError("Insufficient number of peaks")
    return spectrum


def _scale_to_unit_norm(spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
    """
    Scale fragment ion intensities to unit norm.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum for which to scale the fragment ion intensities.

    Returns
    -------
    sus.MsmsSpectrum
        The spectrum with scaled fragment ion intensities.
    """
    spectrum._inner._intensity = spectrum.intensity / np.linalg.norm(
        spectrum.intensity
    )
    return spectrum
