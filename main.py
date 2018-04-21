from tkinter import filedialog, Tk

import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv, read_hdf
from scipy.constants import mu_0, physical_constants, h, m_e
from scipy.odr import Model, RealData, ODR
from scipy.optimize import curve_fit
from seaborn import set_style
from uncertainties import ufloat

from range_selector import RangeTool

set_style("whitegrid")
# set_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
plt.switch_backend('QT5Agg')
mu_B = physical_constants['Bohr magneton']
electron_g = physical_constants['electron g factor']


def electron_lande(J, S, L, gL=1 - (m_e / 2.2069468e-25), gS=electron_g[0]):
    """

    :param J: Total angular momentum of electron.
    :param S: Spin angular momentum of electron.
    :param L: Orbital angular momentum of electron.
    :param gL: Electron orbital g-factor.
    :param gS: Electron spin g-factor.
    :return gJ: Landé g-factor.
    """
    term1 = gL * ((J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)))
    term2 = gS * ((J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1)))
    return term1 + term2


def hyperfine_lande(F, I, J, gJ, gI=-0.00039885395):
    """

    :param F: Coupled nuclear and total electron angular momentum.
    :param I: Nuclear spin.
    :param J: Total angular momentum of electron.
    :param gJ: Landé g-factor.
    :param gI: Nuclear g-factor.
    :return gF: Hyperfine Landé g-factor.
    """
    term1 = gJ * ((F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1)))
    term2 = gI * ((F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1)))
    return term1 + term2


print(hyperfine_lande(F=3, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)))


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
    return (8 / (125 ** (1 / 2))) * (mu_0 * 50 * I) / R


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
    initial = 2e-5, 3e-5, 0.1
    bounds = [[1e-5, 1e-5, 0], [15e-5, 15e-5, 1]]
    uncerts = [10000 for x in range(0, len(data))]
    popt, pcov = curve_fit(freq_as_curr, data['I'], data['f'], p0=initial, bounds=bounds, sigma=uncerts,
                           absolute_sigma=True, method='trf')
    errors = np.diag(pcov)
    popu = [ufloat(popt[0] * 10000, errors[0] * 10000), ufloat(popt[1] * 10000, errors[1] * 10000),
            ufloat(popt[2], errors[2])]
    # print(pcov)
    print('\n')
    print('-------------------------------------------------------------------')
    print("\tEarth's magnetic flux density (∥): \t {:.6f} G".format(popu[0]))
    print("\tEarth's magnetic flux density (⟂): \t {:.6f} G".format(popu[1]))
    print('\tLandé g-factor: \t \t \t \t \t {:.6f}'.format(popu[2]))
    print('-------------------------------------------------------------------')
    print('\n')
    max_x = np.max(data['I'])
    max_y = np.max(data['f'])
    plot_vals = np.linspace(-max_x, max_x, 1000)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', alpha=0.5)
    ax.axvline(x=0, color='k', alpha=0.5)
    plt.xlabel('Current (A)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Hyperfine energy splitting as a function of Helmholtz coil current')
    figure2, = ax.plot(data['I'], data['f'], 'x', ms=15, mew=1.5, antialiased=True)
    ax.plot(plot_vals, freq_as_curr(plot_vals, *popt), antialiased=True, lw=3)
    Sel = RangeTool(data['I'], data['f'], figure2, ax, 'Thing')
    plt.xlim((-max_x * 1.15, max_x * 1.15))
    plt.ylim((0, max_y * 1.15))
    fullscreen()
    plt.show()


def vectorized_freq_as_curr(P, I):
    """
    :param I:
    :param P: Vectorized Parameters.
            P[0]: Hyperfine Landé g-factor.
            P[1]: Earth's magnetic flux density component parallel to Helmholtz coil field.
            P[2]: Earth's magnetic flux density component perpendicular to Helmholtz coil field.
    :return:
    """
    delta_m = 1.0
    R = 0.31
    return (P[0] * mu_B[0] * delta_m / h) * np.sqrt(
        ((8.0 / (np.sqrt(125.0))) * ((mu_0 * 50.0 * I) / R) + P[1]) ** 2 + P[2] ** 2)


def vectorized_freq_as_curr_fitting():
    """
    Curve fitting to find Landé g-factor and Earth's magnetic flux density.
    """
    data, filename = pick_dat(['f', 'I', 'f_uncert', 'I_uncert'], 'RDAT')
    model = Model(vectorized_freq_as_curr)
    mydata = RealData(x=data['I'], y=data['f'], sx=data['I_uncert'], sy=data['f_uncert'])
    myodr = ODR(mydata, model, beta0=[0.25, 2.e-5, 3.e-5], maxit=10000, sstol=0.000000001)
    myoutput = myodr.run()
    myoutput.pprint()
    opt_vals = myoutput.beta
    opt_errs = myoutput.sd_beta
    popu = [ufloat(opt_vals[0], opt_errs[0]), ufloat(opt_vals[1] * 10000, opt_errs[1] * 10000),
            ufloat(opt_vals[2] * 10000,
                   opt_errs[2] * 10000)]
    print('\n')
    print('-------------------------------------------------------------------')
    print('\tLandé g-factor: \t \t \t \t \t {:.6f}'.format(popu[0]))
    print("\tEarth's magnetic flux density (∥): \t {:.6f} G".format(popu[1]))
    print("\tEarth's magnetic flux density (⟂): \t {:.6f} G".format(popu[2]))
    print('-------------------------------------------------------------------')
    print('\n')
    max_x = np.max(data['I'])
    max_y = np.max(data['f'])
    plot_vals = np.linspace(-max_x, max_x, 1000)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', alpha=0.5)
    ax.axvline(x=0, color='k', alpha=0.5)
    plt.xlabel('Current (A)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Hyperfine energy splitting as a function of Helmholtz coil current')
    figure2, = ax.plot(data['I'], data['f'], 'x', ms=15, mew=1.5, antialiased=True)
    ax.plot(plot_vals, vectorized_freq_as_curr(I=plot_vals, P=myoutput.beta), antialiased=True, lw=3)
    Sel = RangeTool(data['I'], data['f'], figure2, ax, 'Thing')
    plt.xlim((-max_x * 1.15, max_x * 1.15))
    plt.ylim((0, max_y * 1.15))
    fullscreen()
    plt.show()


vectorized_freq_as_curr_fitting()
