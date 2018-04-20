from tkinter import filedialog, Tk

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv, read_hdf
from scipy.constants import mu_0, physical_constants, h
from scipy.optimize import curve_fit
from seaborn import set_style
from uncertainties import ufloat

from range_selector import RangeTool

set_style("whitegrid")
plt.switch_backend('QT5Agg')
mu_B = physical_constants['Bohr magneton']


def fullscreen():
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()


def pick_dat(cols, initdir='RDAT', title="Select file"):
    """
    Data reader that is called within many other functions.
    :param initdir: This is the directory that the function will open by default to look for the data (.csv or .h5).
    :param title: The message to display at the top of the dialogue box.
    :param cols: Headers to give to the data.
    :return: Pandas DataFrame with headers that contains the selected data.
    """
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="C:\\Users\Josh\IdeaProjects\OpticalPumping\{}".format(
            initdir),
            title=title)
    filename_parts = root.filename.split('/')[-1]
    if 'csv' in root.filename:
        data = read_csv(root.filename, names=cols, engine='c')
        return data, filename_parts
    elif 'h5' in root.filename:
        data = read_hdf(root.filename, 'table', names=cols, engine='c')
        return data, filename_parts
    else:
        print('Unexpected file type. Choose either a .csv or .h5 file.')


def helmholtz(I, R):
    """

    :param I: Current going into the Helmholtz coils.
    :param n: Number of turns in the coils.
    :param R: Radius of the coils.
    :return: Magnetic field strength at the midpoint between the two coils.
    """
    return ((4 / 5) ** (3 / 2)) * (mu_0 * 50 * I) / R


# TODO: Consider posibility of adjusting Pythargoras' theorem to a 3D form.
def mag_field_comps(B_coil, B_parr, B_perp):
    """

    :param B_coil: Magnetic field due to the Helmholtz coils.
    :param B_parr: Component of Earth's magnetic field parallel to B_coil.
    :param B_perp: Component of Earth's magnetic field perpendicular to B_coil.
    :return:
    """
    return np.sqrt((B_coil + B_parr) ** 2 + B_perp ** 2)


def zeeman(B_tot, g_f):
    """

    :param B_tot: Total magnetic field strength.
    :param g_f: Landé g-factor.
    :param delta_m: Change in z-axis component of hyperfine coupled angular momentum F.
    :return: Energy spacing between levels split by m_F levels.
    """
    return g_f * mu_B[0] * B_tot


def freq_as_curr(I, B_parr_temp, B_perp_temp, g_f_temp):
    """

    :param I: Current going into the Helmholtz coils.
    :param B_parr_temp: Component of Earth's magnetic field parallel to B_coil.
    :param B_perp_temp: Component of Earth's magnetic field perpendicular to B_coil.
    :param g_f_temp: Landé g-factor.
    :return: Frequency separation of Zeeman split levels.
    """
    B_coil_temp = helmholtz(I, 0.31)
    B_tot_temp = mag_field_comps(B_coil_temp, B_parr_temp, B_perp_temp)
    shift = zeeman(B_tot_temp, g_f_temp)
    return shift / h


# TODO: Improve uncertainties.
def freq_as_curr_fitting():
    """
    Curve fitting to find Landé g-factor and Earth's magnetic flux density.
    """
    data, filename = pick_dat(['f', 'I'], 'RDAT')
    initial = 2e-5, 3e-5, -0.0004
    bounds = [[0.1e-5, 0.1e-5, -0.00045], [105e-5, 105e-5, 2.2]]
    uncerts = [1000 for x in range(0, len(data))]
    popt, pcov = curve_fit(freq_as_curr, data['I'], data['f'], p0=initial, bounds=bounds, sigma=uncerts,
                           absolute_sigma=True, method='trf')
    errors = np.diag(pcov)
    popu = [ufloat(popt[0], errors[0]), ufloat(popt[1], errors[1]), ufloat(popt[2], errors[2])]
    print(pcov)
    print('\n')
    print('-------------------------------------------------------------------')
    print("\tEarth's magnetic flux density (∥): \t {:.6e}".format(popu[0]))
    print("\tEarth's magnetic flux density (⟂): \t {:.6e}".format(popu[1]))
    print('\tLandé g-factor: \t \t \t \t \t {:.6f}'.format(popu[2]))
    print('-------------------------------------------------------------------')
    print('\n')
    max_x = np.max(data['I'])
    max_y = np.max(data['f'])
    plot_vals = np.linspace(-max_x, max_x, 1000)
    fig, ax = plt.subplots()
    plt.xlim([-max_x * 1.05, max_x * 1.05])
    plt.ylim([-max_x * 1.05, max_y * 1.05])
    plt.xlabel('Current (A)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Hyperfine energy splitting as a function of Helmholtz coil current')
    figure2, = ax.plot(data['I'], data['f'], 'x', ms=10, antialiased=True)
    # ax.plot(data['I'], freq_as_curr(data['I'], *popt), antialiased=True)
    ax.plot(plot_vals, freq_as_curr(plot_vals, *popt), antialiased=True)
    Sel = RangeTool(data['I'], data['f'], figure2, ax, 'Thing')
    fullscreen()
    plt.show()


freq_as_curr_fitting()
