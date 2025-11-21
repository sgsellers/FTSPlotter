import warnings
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np

from adjustText import adjust_text

def fts_plot(cwvl: float, wvl_range: float, lines: bool=True) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    For a given wavelength range, plot the FTS reference spectrum

    Parameters
    ----------
    cwvl : float
        Central wavelength in angstroms
    wvl_range : float
        Range around central wavelength
    lines : bool
        If true, overplots line names

    Returns
    -------
    fts_wave : numpy.ndarray
    fts_spec : numpy.ndarray
    line_centers : list
    line_names : list
    """

    def read_data(path, fname) -> np.ndarray:
        with resources.path(path, fname) as df:
            return np.load(df)

    def find_nearest(array, value) -> int:
        return np.abs(np.nan_to_num(array) - value).argmin()

    atlas_angstrom = read_data('FTSPlotter.atlas', 'FTS1984_296-1300nm_Wavelengths.npy')
    atlas_spectrum = read_data('FTSPlotter.atlas', 'FTS1984_296-1300nm_Atlas.npy')

    idx_lo = find_nearest(atlas_angstrom, cwvl - wvl_range / 2) - 5
    idx_hi = find_nearest(atlas_angstrom, cwvl + wvl_range / 2) + 5

    wave = atlas_angstrom[idx_lo:idx_hi]
    spec = atlas_spectrum[idx_lo:idx_hi]

    if lines:
        line_centers = read_data('FTSPlotter.atlas', 'moore_linecenters.npy')
        line_names = read_data('FTSPlotter.atlas', 'moore_linenames.npy')
        line_selection = (line_centers > wave[0]) & (line_centers < wave[-1])
        centers = line_centers[line_selection]
        names = line_names[line_selection]
    else:
        centers = []
        names = []

    if len(centers) > 40:
        warnings.warn("Many spectral lines ({0}) within the specified range! Plot will be messy.".format(len(centers)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wave, spec, c='C1')
    ax.set_xlim(wave[0], wave[-1])
    ax.set_ylim(0, 1.2)
    texts = []
    num_na = 0
    for i in range(len(centers)):
        if 'N/A' not in names[i]:
            ax.axvline(centers[i], linestyle='--', c='C{0}'.format(i%10))
            texts.append(ax.text(centers[i], 1.1, names[i], c='C{0}'.format(i%10), weight='bold'))
        else:
            ax.plot(
                [centers[i], centers[i]],
                [0, 0.25],
                linestyle='--',
                c='C3'
            )
            num_na += 1
    if num_na > 0:
        fig.suptitle("Short Lines = Unknown/Insignificant", weight='bold')
    adjust_text(texts, ax=ax)#, arrowprops=dict(arrowstyle='-', color='gray', lw=1))
    plt.show()

    return wave, spec, centers, names


def fts_window(wavemin, wavemax, atlas='FTS', norm=True, lines=False):
    """
    For a given wavelength range, return the solar reference spectrum within that range.

    :param wavemin: float
        Blue end of the wavelength range
    :param wavemax: float
        Red end of the wavelength range
    :param atlas: str
        Which atlas to use. Currently accepts "Wallace" and "FTS"
        Wallace uses the 2011 Wallace updated atlas
        FTS uses the 1984 FTS atlas
    :param norm: bool
        If False, and the atlas is set to "FTS", will return the solar irradiance.
        This includes the blackbody curve, etc.
    :param lines: bool
        If True, returns additional arrays denoting line centers and names
        within the wavelength range.
    :return wave: array-like
        Array of wavelengths
    :return spec: array-like
        Array of spectral values
    :return line_centers: array-like, optional
        Array of line center positions
    :return line_names: array-like, optional
        Array of line names
    """

    def read_data(path, fname) -> np.array:
        with resources.path(path, fname) as df:
            return np.load(df)

    if atlas.lower() == 'wallace':
        if (wavemax <= 5000.) or (wavemin <= 5000.):
            atlas_angstroms = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_290-1000nm_Wavelengths.npy')
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_290-1000nm_Observed.npy')
        else:
            atlas_angstroms = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_500-1000nm_Wavelengths.npy')
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'Wallace2011_500-1000nm_Corrected.npy')
    else:
        atlas_angstroms = read_data('ssosoft.spectral.FTS_atlas', 'FTS1984_296-1300nm_Wavelengths.npy')
        if norm:
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'FTS1984_296-1300nm_Atlas.npy')
        else:
            warnings.warn("Using solar irradiance (i.e., not normalized)")
            atlas_spectrum = read_data('ssosoft.spectral.FTS_atlas', 'FTS1984_296-1300nm_Irradiance.npy')
            atlas_spectrum *= 462020 # Conversion to erg/cm2/s/nm
            atlas_spectrum /= 10 # Conversion to erg/cm2/s/Angstrom

    idx_lo = find_nearest(atlas_angstroms, wavemin) - 5
    idx_hi = find_nearest(atlas_angstroms, wavemax) + 5

    wave = atlas_angstroms[idx_lo:idx_hi]
    spec = atlas_spectrum[idx_lo:idx_hi]

    if lines:
        line_centers_full = read_data(
            'FTS_atlas',
            'RevisedMultiplet_Linelist_2950-13200_CentralWavelengths.npy'
        )
        line_names_full = read_data(
            'FTS_atlas',
            'RevisedMultiplet_Linelist_2950-13200_IonNames.npy'
        )
        line_selection = (line_centers_full < wavemax) & (line_centers_full > wavemin)
        line_centers = line_centers_full[line_selection]
        line_names = line_names_full[line_selection]
        return wave, spec, line_centers, line_names
    else:
        return wave, spec
