from time import gmtime, strftime
from tkinter import filedialog, Tk
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.models import GaussianModel, LinearModel
from pandas import read_csv, read_hdf, DataFrame
from scipy import fftpack
from scipy.constants import mu_0, physical_constants, h
# from scipy.odr import Model, RealData, ODR
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from seaborn import set_style
from sympy import diff, Symbol, latex
from uncertainties import ufloat

from range_selector import RangeTool

e = 2.7182818
today = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
set_style("whitegrid")
# set_palette("Set1")
plt.switch_backend('QT5Agg')
mu_B = physical_constants['Bohr magneton']
electron_g = physical_constants['electron g factor']
m_e = physical_constants['electron mass']
m_cs = [2.2069468e-25, 3.3210778e-34]
g_I = [-0.00039885395, 0.0000000000052]
lab_field_NOAA = [48585.7 * 0.00001, 152 * 0.00001]
formatted_NOAA = ufloat(lab_field_NOAA[0], lab_field_NOAA[1])


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def electron_lande(J, S, L, gL=(1 - (m_e[0] / (m_cs[0] - (55 * m_e[0])))), gS=-electron_g[0]):
    """

    :param J: Total angular momentum of electron.
    :param S: Spin angular momentum of electron.
    :param L: Orbital angular momentum of electron.
    :param gL: Electron orbital g-factor.
    :param gS: Electron spin g-factor.
    :return gJ: Landé g-factor.
    """
    term1 = gL * (((J * (J + 1)) - (S * (S + 1)) + (L * (L + 1))) / (2 * (J * (J + 1))))
    term2 = gS * (((J * (J + 1)) + (S * (S + 1)) - (L * (L + 1))) / (2 * (J * (J + 1))))
    return term1 + term2


def error_prop_sympy():
    g = Symbol('g_F')
    mu1 = Symbol('mu_B')
    h = Symbol('h')
    mu2 = Symbol('mu_0')
    N = Symbol('N')
    I = Symbol('I')
    R = Symbol('R')
    B1 = Symbol('B_parr')
    B2 = Symbol('B_perp')
    v = (g * mu1 / h) * (
            ((8.0 / (125.0 ** (1 / 2))) * ((mu2 * N * I) / R) + B1) ** 2 + B2 ** 2) ** (1 / 2)
    print(latex(diff(v, B1)))


def hyperfine_lande(F, I, J, gJ, gI=g_I[0]):
    """

    :param F: Coupled nuclear and total electron angular momentum.
    :param I: Nuclear spin.
    :param J: Total angular momentum of electron.
    :param gJ: Landé g-factor.
    :param gI: Nuclear g-factor.
    :return gF: Hyperfine Landé g-factor.
    """
    term1 = gJ * (((F * (F + 1)) - (I * (I + 1)) + (J * (J + 1))) / (2 * (F * (F + 1))))
    term2 = gI * (((F * (F + 1)) + (I * (I + 1)) - (J * (J + 1))) / (2 * (F * (F + 1))))
    return term1 + term2


def hyperfine_lande_uncert(F, I, J, S, L):
    term1 = (((J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1))) * (m_e[2] / m_cs[0])) ** 2
    term2 = (((J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1))) * (m_cs[1] * m_e[0] / m_cs[0] ** 2)) ** 2
    term3 = (((J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))) * electron_g[2]) ** 2
    term4 = (((F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))) * g_I[1]) ** 2
    return np.sqrt(term1 + term2 + term3 + term4)


gF3 = [hyperfine_lande(F=3, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)),
       hyperfine_lande_uncert(F=3, I=7 / 2, J=1 / 2, S=1 / 2, L=0)]
gF2 = [hyperfine_lande(F=2, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)),
       hyperfine_lande_uncert(F=2, I=7 / 2, J=1 / 2, S=1 / 2, L=0)]
gF4 = [hyperfine_lande(F=4, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)),
       hyperfine_lande_uncert(F=4, I=7 / 2, J=1 / 2, S=1 / 2, L=0)]

formatted_gf4 = ufloat(gF4[0], gF4[1])
formatted_gf3 = ufloat(gF3[0], gF3[1])
formatted_gf2 = ufloat(gF2[0], gF3[1])
print('g_(F=4):', formatted_gf4)
print('g_(F=3):', formatted_gf3)


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


def range_to_list(smooth=False):
    """
    This function is used to create an array of values from a dataset that's limits are given by a list lower and
    upper limits. THIS IS CONFIGURED FOR MY COMPUTER, CHANGE THE DIRECTORY TO USE.
    """
    dat1, filename1 = pick_dat(['V', 'T'], "Sweep_dat", "Select dataset to draw from")
    dat2 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_ranges\\{}".format(filename1),
                    names=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
    if smooth:
        win_len = int(len(dat1) / 10)
        if win_len % 2 == 0:
            win_len += 1
        dat1['T'] = savgol_filter(dat1['T'], win_len, 1)
    xrange = []
    yrange = []
    xranges = {}
    yranges = {}
    x_append = xrange.append
    y_append = yrange.append
    for o in range(0, len(dat2)):
        x_append((dat1['V'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
        y_append((dat1['T'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
    for o in range(0, len(xrange)):
        xranges[o] = xrange[o]
        yranges[o] = yrange[o]
    return xranges, yranges, xrange, yrange, filename1, dat1


def echo_as_T2(t, M0, T2, m, m1, c, ph):
    """

    :param t:
    :param M0: Initial magnetization in z direction.
    :param T2: Spin-spin relaxation time.
    :param c: Intercept to compensate for DC-offset.
    :param ph: Phase difference.
    :return: Magnetization in the xy-plane.
    """
    # Old form:
    return M0 * (np.exp(-((t - ph) / T2))) - m1 * t ** 2 + m * t + c
    # return M0 * (np.exp(-(t / T2) + ph)) + c


def Rabi_1(t, A, o_ab, g_ab, o, w_ba, w, g, c1, c0, p):
    num = (o_ab ** 1) * np.exp(-g_ab * t / 2) * (np.sin(t * o / 2 + p)) ** 1
    denom = np.sqrt(((w_ba - w) ** 2) + ((g / 2) ** 2) + o_ab ** 2)
    return A * (num / denom) + + c1 * t + c0


def Rabi_2(t, A, o, g, p, c):
    o_g = np.sqrt(o ** 2 + (g / 4) ** 2)
    term1 = np.cos(o_g * (t + p) + np.sin(o_g * (t + p)) * (o ** 2 - g ** 2 / 4) / (g * o_g))
    term2 = 1 - np.exp(-(3 * g * (t + p) / 4)) * term1
    term3 = -1 + (o ** 2) / (o ** 2 + (g ** 2) / 2) * term2
    return A * term3 + c


def Rabi_3(t, O, w, A, c, p1, p2, m):
    return -A * np.sin(2 * O * t + p1) * np.sin(w * t + p2) + m * t + c


def damped_sine(t, A, o, g, p, c):
    return A * np.exp(-g * t) * np.sin(t * o + p) ** 2 + c


def helmholtz(I, R):
    """

    :param I: Current going into the Helmholtz coils.
    :param n: Number of turns in the coils.
    :param R: Radius of the coils.
    :return: Magnetic field strength at the midpoint between the two coils.
    """
    return (8 / (125 ** (1 / 2))) * (mu_0 * 50 * I) / R


def helmholtz_uncert(I, I_uncert):
    C = 8 / (125 ** (1 / 2))
    N = 50
    R = 0.2915
    N_uncert = 1
    R_uncert = 0.02
    term_N = ((C * mu_0 * I / R) * N_uncert) ** 2
    term_I = ((C * mu_0 * N / R) * I_uncert) ** 2
    term_R = ((-C * mu_0 * N * I / (R ** 2)) * R_uncert) ** 2
    return np.sqrt(term_N + term_I + term_R)
    

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
    R = 0.29
    return (P[0] * mu_B[0] * delta_m / h) * np.sqrt(
            ((8.0 / (np.sqrt(125.0))) * ((mu_0 * 50.0 * I) / R) + P[1]) ** 2 + P[2] ** 2)


# TODO: Improve uncertainties.
def freq_as_curr_fitting():
    """
    Curve fitting to find Landé g-factor and Earth's magnetic flux density.
    """
    data, filename = pick_dat(['f', 'I', 'f_uncert', 'I_uncert'], 'RDAT')
    initial = 2e-5, 3e-5, 0.1
    bounds = [[1e-5, 1e-5, 0], [15e-5, 15e-5, 1]]
    uncerts = [10000 for x in range(0, len(data))]
    popt, pcov = curve_fit(freq_as_curr, data['I'], data['f'], p0=initial, bounds=bounds, sigma=uncerts,
                           absolute_sigma=True, method='trf')
    errors = np.diag(pcov)
    popu = [ufloat(popt[0] * 10000, errors[0] * 10000), ufloat(popt[1] * 10000, errors[1] * 10000),
            ufloat(popt[2], errors[2])]
    chisq = np.sum(((data['f'] - freq_as_curr(data['I'], *popt)) ** 2) / data['f'])
    net_mag = np.sqrt((popt[0] * 10000) ** 2 + (popt[1] * 10000) ** 2)
    net_mag_err = np.sqrt(((popt[0] * errors[0] * 100000000) ** 2 / ((popt[0] * 10000) ** 2 + (popt[1] * 10000) ** 2)) +
                          ((popt[1] * errors[1] * 100000000) ** 2 / ((popt[0] * 10000) ** 2 + (popt[1] * 10000) ** 2)))
    popu_calc = [ufloat(net_mag, net_mag_err)]
    print('\n')
    print('-------------------------------------------------------------------')
    print("\tEarth's magnetic flux density (∥): \t \t {:.6f} G".format(popu[0]))
    print("\tEarth's magnetic flux density (⟂): \t \t {:.6f} G".format(popu[1]))
    print('\n')
    print("\tEarth's magnetic flux density: \t \t \t {:.6f} G".format(popu_calc[0]))
    print("\tNOAA Earth's magnetic flux density: \t {:.6f} G".format(formatted_NOAA))

    print('\n')
    print('\tLandé g-factor: \t \t \t \t \t \t {:.6f}'.format(popu[2]))
    print('\tCalculated Landé g-factor: \t \t \t \t {:.6f}'.format(formatted_gf3))

    print('\n')
    print('\tReduced chi-sq: \t \t \t \t \t \t {:.2f}'.format(chisq / (len(data['I']) - len(popt))))
    print('-------------------------------------------------------------------')
    print('\n')
    max_x = np.max(data['I'])
    max_y = np.max(data['f'])
    plot_vals = np.linspace(float(-max_x), max_x, 1000)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', alpha=0.5)
    ax.axvline(x=0, color='k', alpha=0.5)
    plt.xlabel('Current (A)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Hyperfine energy splitting as a function of Helmholtz coil current')
    figure2, = ax.plot(data['I'], data['f'], 'o', markerfacecolor="None", color='#050505',
                       mew=1.4, ms=7, antialiased=True, label='Data')
    ax.plot(plot_vals, freq_as_curr(plot_vals, *popt), antialiased=True, lw=2.5, label='TRF Fit', color='k')
    Sel = RangeTool(data['I'], data['f'], figure2, ax, 'Thing')
    plt.xlim((-max_x * 1.15, max_x * 1.15))
    plt.ylim((0, max_y * 1.15))
    fullscreen()
    plt.legend()
    plt.savefig("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\FreqAsCurr_{}.png".format(
            today), dpi=600)
    plt.show()


def vectorized_freq_as_curr_fitting():
    """
    Non-linear curve fitting using the vectorized function (that allows for both x and y uncertainty to be
    compensated for) to find Landé g-factor and Earth's magnetic flux density.
    """
    data, filename = pick_dat(['f', 'I', 'f_uncert', 'I_uncert'], 'RDAT')
    model = Model(vectorized_freq_as_curr)
    x_err = helmholtz_uncert(data['I'], data['I_uncert'])
    mydata = RealData(x=data['I'], y=data['f'], sx=data['I_uncert'], sy=data['f_uncert'])
    myodr = ODR(mydata, model, beta0=[0.25, 2.e-5, 3.e-5], maxit=10000)
    myoutput = myodr.run()
    opt_vals = myoutput.beta
    opt_errs = myoutput.sd_beta
    popu = [ufloat(opt_vals[0], opt_errs[0]), ufloat(opt_vals[1] * 10000, opt_errs[1] * 10000),
            ufloat(opt_vals[2] * 10000,
                   opt_errs[2] * 10000)]
    net_mag = np.sqrt((opt_vals[1] * 10000) ** 2 + (opt_vals[2] * 10000) ** 2)
    net_mag_err = np.sqrt(
            ((opt_vals[1] * opt_errs[1] * 100000000) ** 2 / ((opt_vals[1] * 10000) ** 2 + (opt_vals[2] * 10000) ** 2)) +
            ((opt_vals[2] * opt_errs[2] * 100000000) ** 2 / ((opt_vals[1] * 10000) ** 2 + (opt_vals[2] * 10000) ** 2)))
    popu_calc = [ufloat(net_mag, net_mag_err)]
    mag_flux_overlap = (abs(popu_calc[0].n - formatted_NOAA.n) / (popu_calc
                                                                  [0].std_dev + formatted_NOAA.std_dev))
    lande_overlap = (abs(popu[0].n - formatted_gf4.n) / (popu[0].std_dev + formatted_gf4.std_dev))
    B_tot = mag_field_comps(helmholtz(data['I'], 0.2914), opt_vals[1], opt_vals[2])
    B_coil = helmholtz(data['I'], 0.2914)
    print(B_tot * 10000)
    print('\n')
    print('-------------------------------------------------------------------')
    print("\tEarth's magnetic flux density (∥): \t \t {:.4f} G".format(popu[1]))
    print("\tEarth's magnetic flux density (⟂): \t \t {:.4f} G".format(popu[2]))
    print('\n')
    print("\tEarth's magnetic flux density: \t \t \t {:.4f} G".format(popu_calc[0]))
    print("\tNOAA Earth's magnetic flux density: \t {:.4f} G".format(formatted_NOAA))
    print('\tStd. Dev. Separation: \t \t \t \t \t {:.2f}'.format(mag_flux_overlap))
    print('\n')
    print('\tLandé g-factor: \t \t \t \t \t \t {:.4f}'.format(popu[0]))
    print('\tCalculated Landé g-factor: \t \t \t \t {:.15f}'.format(formatted_gf4))
    print('\tStd. Dev. Separation: \t \t \t \t \t {:.2f}'.format(lande_overlap))
    print('\n')
    print('\tReduced chi-sq: \t \t \t \t \t \t {:.2f}'.format(myoutput.res_var / (len(data['I']) - len(opt_vals))))
    print('-------------------------------------------------------------------')
    print('\n')
    max_x = np.max(data['I'])
    min_x = np.min(data['I'])
    max_y = np.max(data['f'])
    plot_vals = np.linspace(min_x, max_x, 1000)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', alpha=0.5)
    ax.axvline(x=0, color='k', alpha=0.5)
    plt.xlabel('Current (A)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Hyperfine energy splitting as a function of Helmholtz coil current')
    figure2, = ax.plot(data['I'], data['f'], 'o', markerfacecolor="None", color='#050505',
                       mew=1.4, ms=7, antialiased=True, label='Data')
    figure3, = ax.plot(plot_vals, vectorized_freq_as_curr(I=plot_vals, P=myoutput.beta), antialiased=True, lw=2.5,
                       label='ODR Fit', color='k')
    plt.errorbar(data['I'], data['f'], xerr=data['I_uncert'], yerr=data['f_uncert'], fmt='none', ecolor='#050505',
                 label=None)
    Sel = RangeTool(plot_vals, vectorized_freq_as_curr(I=plot_vals, P=myoutput.beta), figure3, ax, 'Thing')
    plt.xlim((-max_x * 1.15, max_x * 1.15))
    plt.ylim((0, max_y * 1.15))
    fullscreen()
    plt.legend()
    plt.savefig("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\VectorizedFreqAsCurr_{}.png".format(
            today), dpi=600)
    plt.show()


def read_scan():
    data, filename = pick_dat(['f', 'RT'], 'Sweep_dat')
    meanfreq = np.mean(data['f'])
    fig, ax = plt.subplots()
    figure1, = ax.plot(data['f'], data['RT'], '.', markerfacecolor="None", color='#050505',
                       mew=1.4, ms=1, antialiased=True, label='Data')
    window = int(len(data) / 20)
    if window % 2 == 0:
        window += 1
    figure2, = ax.plot(data['f'], savgol_filter(data['RT'], window, 2), lw=2)
    if filename.endswith('.csv'):
        name = filename[:-4]
    # Thing = RangeTool(data['f'], data['RT'], figure1, ax, name)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Variance (a.u)', fontsize=14)
    plt.xlim([np.min(data['f']), np.max(data['f'])])
    ax.axes.tick_params(labelsize=12)
    plt.title('Relative light transmission for a given Helmholtz current as a function of frequency')
    fullscreen()
    plt.savefig("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\Sweep_plot_{:.3}Hz_{}.png".format(
            meanfreq, today), dpi=600)
    plt.show()


def fit_gauss(graph=False, cdf=False, residuals=False):
    xranges, yranges, xrange, yrange, filename1, dat1 = range_to_list()
    FitVals = DataFrame(columns=['Sigma', 'Center', 'Amplitude', 'FWHM', 'Height', 'Intercept', 'Slope', 'ChiSq',
                                 'RedChiSq', 'Akaike', 'Bayesian'])
    for i in range(0, len(xranges)):
        mdl = GaussianModel()
        params = mdl.guess(data=yranges[i], x=xranges[i])
        result = mdl.fit(yranges[i], params, x=xranges[i])
        print(result.fit_report())
        FitVals.at[i, 'Sigma'] = ufloat(result.params['sigma'].value, result.params['sigma'].stderr)
        FitVals.at[i, 'Center'] = ufloat(result.params['center'].value, result.params['center'].stderr)
        FitVals.at[i, 'Amplitude'] = ufloat(result.params['amplitude'].value, result.params['amplitude'].stderr)
        FitVals.at[i, 'FWHM'] = ufloat(result.params['fwhm'].value, result.params['fwhm'].stderr)
        FitVals.at[i, 'Height'] = ufloat(result.params['height'].value, result.params['height'].stderr)
        FitVals.at[i, 'ChiSq'] = result.chisqr
        FitVals.at[i, 'RedChiSq'] = result.redchi
        FitVals.at[i, 'Akaike'] = result.aic
        FitVals.at[i, 'Bayesian'] = result.bic
        if graph:
            plt.plot(xranges[i], yranges[i], '.', markerfacecolor="None", color='#050505',
                     mew=1.4, ms=1, antialiased=True, label='Data from frequency sweep')
            plt.plot(xranges[i], result.best_fit, lw=2, label='Gaussian + Line fit')
            plt.legend()
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Variance (a.u)')
            fullscreen()
            plt.show()
        if cdf:
            values, base = np.histogram(yranges[i], bins='auto')
            cumulative = np.cumsum(values)
            plt.plot(base[:-1], cumulative)
            plt.plot(base[:-1], values)
            fullscreen()
            plt.show()
        if residuals:
            values, base = np.histogram(result.residual, bins='fd')
            cumulative = np.cumsum(values)
            # plt.plot(xranges[i], result.residual)
            # plt.plot(base[:-1], cumulative, '.')
            mdl2 = GaussianModel()
            params2 = mdl2.guess(data=values, x=base[:-1])
            model2 = mdl2
            result2 = model2.fit(values, params2, x=base[:-1])
            plt.plot(base[:-1], values, '.')
            plt.plot(base[:-1], result2.best_fit)
            plt.xlabel('Residuals')
            plt.ylabel('Counts')
            fullscreen()
            plt.show()


def bulk_fit():
    datafolder = filedialog.askopenfilenames(initialdir="C:\\Users\Josh\IdeaProjects\OpticalPumping",
                                             title="Select data to convert")
    k = -1
    FitVals = DataFrame(columns=['Current', 'Sigma', 'Center', 'Amplitude', 'FWHM', 'Height', 'Intercept',
                                 'Slope', 'ChiSq',
                                 'RedChiSq', 'Akaike', 'Bayesian'])
    rdat = DataFrame(columns=['f', 'I', 'f_uncert', 'I_uncert'])
    for filename in datafolder:
        name_ext = filename.split('/')[-1]
        if 'A' in filename and 'NO' not in filename and 'ODD' not in filename:
            k += 1
            dat1 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_dat\\{}".format(name_ext),
                            names=['f', 'RT'])
            dat2 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_ranges\\{}".format(name_ext),
                            names=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
            xrange = []
            yrange = []
            xranges = {}
            yranges = {}
            x_append = xrange.append
            y_append = yrange.append
            for o in range(0, len(dat2)):
                x_append((dat1['f'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
                y_append((dat1['RT'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
            for o in range(0, len(xrange)):
                xranges[o] = xrange[o]
                yranges[o] = yrange[o]
            for i in range(0, len(xranges)):
                mdl = GaussianModel()
                line = LinearModel()
                params = mdl.guess(data=yranges[i], x=xranges[i])
                params += line.guess(data=yranges[i], x=xranges[i])
                model = mdl + line
                result = model.fit(yranges[i], params, x=xranges[i])
                # print(result.fit_report())
                j = len(FitVals)
                leader = name_ext.split('_')[1]
                print(leader)
                current = leader.split('A')[0]
                curr_uncert_step1 = current.split('A')[0]
                if '-' in curr_uncert_step1:
                    curr_uncert_step1 = curr_uncert_step1.split('-')[1]
                curr_uncert_step2 = curr_uncert_step1.split('.')[-1]
                curr_uncert = '0.'
                for p in range(0, len(curr_uncert_step2)):
                    curr_uncert += '0'
                curr_uncert += '5'
                print(curr_uncert)
                FitVals.at[j, 'Current'] = ufloat(float(current), float(curr_uncert))
                rdat.at[j, 'I'] = float(current)
                rdat.at[j, 'I_uncert'] = float(curr_uncert)
                FitVals.at[j, 'Sigma'] = ufloat(result.params['sigma'].value, result.params['sigma'].stderr)
                FitVals.at[j, 'Center'] = ufloat(result.params['center'].value, result.params['center'].stderr)
                rdat.at[j, 'f'] = result.params['center'].value
                rdat.at[j, 'f_uncert'] = result.params['center'].stderr
                FitVals.at[j, 'Amplitude'] = ufloat(result.params['amplitude'].value, result.params['amplitude'].stderr)
                FitVals.at[j, 'FWHM'] = ufloat(result.params['fwhm'].value, result.params['fwhm'].stderr)
                FitVals.at[j, 'Height'] = ufloat(result.params['height'].value, result.params['height'].stderr)
                FitVals.at[j, 'Intercept'] = ufloat(result.params['intercept'].value, result.params['intercept'].stderr)
                FitVals.at[j, 'Slope'] = ufloat(result.params['slope'].value, result.params['slope'].stderr)
                FitVals.at[j, 'ChiSq'] = result.chisqr
                FitVals.at[j, 'RedChiSq'] = result.redchi
                FitVals.at[j, 'Akaike'] = result.aic
                FitVals.at[j, 'Bayesian'] = result.bic
    FitVals.to_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\FitVals\\FitVals_{}.csv".format(today))
    rdat.sort_values(['I'], inplace=True)
    rdat.to_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\RDAT\\Freq_Curr_{}.csv".format(today), index=False,
                header=False)


def bulk_plot():
    datafolder = filedialog.askopenfilenames(initialdir="C:\\Users\Josh\IdeaProjects\OpticalPumping",
                                             title="Select data for bulk plotting")
    for filename in datafolder:
        name_ext = filename.split('/')[-1]
        if 'NO' or 'ODD' in filename:
            run = name_ext.split('_')[1]
            dat1 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_dat\\{}".format(name_ext),
                            names=['f', 'RT'])
            dat1 = dat1[np.abs(dat1['f'] - dat1['f'].mean()) <= (3 * dat1['f'].std())]
            meanfreq = np.mean(dat1['f'])
            fig, ax = plt.subplots(figsize=(14, 6))
            figure1, = ax.plot(dat1['f'], dat1['RT'], '.', markerfacecolor="None", color='#050505',
                               mew=1.4, ms=1, antialiased=True, label='dat1')
            window = int(len(dat1) / 20)
            if window % 2 == 0:
                window += 1
            figure2, = ax.plot(dat1['f'], savgol_filter(dat1['RT'], window, 2), lw=2)
            if filename.endswith('.csv'):
                name = filename[:-4]
            plt.xlabel('Frequency (Hz)', fontsize=14)
            plt.ylabel('Variance (a.u)', fontsize=14)
            plt.xlim([np.min(dat1['f']), np.max(dat1['f'])])
            ax.axes.tick_params(labelsize=12)
            plt.title('{}'.format(run))
            plt.savefig(
                    "C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\Sweep_plot_{:.3f}kHz_{}.png".format(
                            meanfreq / 1000, today), dpi=600)
            plt.close()


def pick_ranges():
    """
    Tool to read data and present it graphically ready for data ranges, to be used in fitting, to be made. Press tab
    to mark the lower bound, shift to mark the upper bound, delete to remove the last range selected, enter to open a
    dialog box to save the ranges as a .csv file. Exit closes the plot without saving ranges.
    """
    dat, filename = pick_dat(['T', 'V'], 'Sweep_dat', 'Select file to pick ranges in')
    fig, ax = plt.subplots()
    name = filename.split('.csv')[0]
    plt.title('{} Fourier Transformed'.format(name))
    figure, = ax.plot(dat['T'], dat['V'])
    Sel = RangeTool(dat['T'], dat['V'], figure, ax, name)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def echo_fits():
    """
    Fits a Gaussian with a linear background to each of the echo peaks, finds the centroid and top of
    the Gaussian, then fits the echo_as_T2 function to the points given by x=centroid, y=top.
    """
    xrs, yrs, xr, yr, filename, dat1 = range_to_list()
    cents: List[float] = []
    cents_uncert: List[float] = []
    heights: List[float] = []
    heights_uncert: List[float] = []
    for i in range(0, len(xrs)):
        print(xrs, yrs)
        mdl = GaussianModel(prefix='G_')
        lne = LinearModel(prefix='L_')
        params = mdl.guess(yrs[i], x=xrs[i])
        params += lne.guess(yrs[i], x=xrs[i])
        max_y = np.max(yrs[i])
        min_y = np.min(yrs[i])
        max_x = np.max(yrs[i])
        min_x = np.min(yrs[i])
        predicted_slope = (max_y - min_y) / (max_x - min_x)
        # params.add('L_slope', value=predicted_slope, min=predicted_slope * 1.1, max=predicted_slope * 0.9)
        # params.add('L_intercept', value=min_y, min=min_y * 0.9, max=min_y * 1.1)
        # params.add('G_height', value=max_y - min_y, min=(max_y - min_y) * 0.99, max=(max_y - min_y) * 1.05)
        model = mdl + lne
        result = model.fit(yrs[i], params, x=xrs[i], method='leastsq')
        plt.plot(xrs[i], result.best_fit)

        cent: float = result.params['G_center'].value
        amp: float = result.params['G_height'].value
        inter: float = result.params['L_intercept'].value
        grad: float = result.params['L_slope'].value
        height: float = amp + ((cent * grad) + inter)
        heights.append(height)
        cents.append(cent)
        cents_uncert.append(result.params['G_center'].stderr)
        partial_amp = 1
        partial_grad = cent
        partial_x = grad
        partial_inter = 1
        amp_term = partial_amp * result.params['G_height'].stderr
        grad_term = partial_grad * result.params['L_slope'].stderr
        x_term = partial_x * np.mean(np.diff(xrs[i]))
        inter_term = partial_inter * result.params['L_intercept'].stderr
        height_uncert = np.sqrt(amp_term ** 2 + grad_term ** 2 + x_term ** 2 + inter_term ** 2)
        heights_uncert.append(height_uncert)
        plt.plot(xrs[i], yrs[i])
    plt.show()
    heights = np.array(heights)
    cents = np.array(cents)
    maxy = np.max(heights)
    miny = np.min(heights)
    decay_pos = np.where(heights == find_nearest(heights, maxy / e))[0][0]
    decay_pos_time = cents[decay_pos]
    print(heights, cents)
    avg_y_sep = abs(np.mean(np.diff(heights)))
    efit = Model(echo_as_T2)
    param = efit.make_params()
    param.add('M0', value=maxy, min=maxy * 0.8, max=maxy + (avg_y_sep * 2))
    param.add('T2', value=decay_pos_time, min=decay_pos_time * 0.1, max=decay_pos_time * 1.5)
    param.add('c', value=miny * 0.3, min=miny * 0.1, max=miny * 1)
    param.add('ph', value=cents[0] * 0.1, min=0, max=cents[0] * 1)
    result_2 = efit.fit(heights, param, t=cents, method='leastsq', weights=np.sqrt(np.mean(np.diff(dat1['V'])) ** 2 +
                                                                                   np.array(heights_uncert) ** 2) /
                                                                           heights)
    print(result_2.fit_report())
    print('\n', result_2.params.pretty_print(fmt='e', precision=2))
    ax = plt.gca()
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Magnetization (A/m)', fontsize=14)
    xes = np.linspace(np.min(cents), np.max(cents), 100)
    y = efit.eval(t=xes, params=result_2.params)
    plt.plot(xes, y, antialiased=True)
    # plt.plot(cents, heights, 'x', ms=8, color='k')
    # plt.plot(dat1['V'], dat1['T'], lw=2, antialiased=True,
    #          color='#4a4a4a', zorder=1)
    plt.title(filename)
    # plt.xlim(left=0, right=np.max(cents) * 1.1)
    # plt.ylim(bottom=0, top=result_2.params['M0'].value * 1.3)
    # plt.axhline(result_2.params['M0'].value, color='k', ls='--', alpha=0.7, lw=1, zorder=2)
    # plt.axhline(result_2.params['M0'].value / e, color='k', ls='--', alpha=0.7, lw=1, zorder=2)
    plt.text(0.9, 0.9, "T_1: {:.4f} s".format(result_2.params['T2'].value), horizontalalignment='center',
             verticalalignment="center",
             transform=ax.transAxes,
             bbox={'pad': 8, 'fc': 'w'}, fontsize=14)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=13)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def simple_echo_fits():
    """
    Takes the highest point of each echo and fits the echo_as_T2 function to those points.
    """
    xrs, yrs, xr, yr, filename, dat1 = range_to_list()
    length = len(yrs)
    max_y = [np.max(yrs[i]) for i in range(length)]
    max_y_loc = [np.where(yrs[i] == max_y[i])[0][0] for i in range(length)]
    cents = [xrs[i][max_y_loc[i]] for i in range(length)]
    heights = max_y
    # TODO: Find a better value for the uncertainty on y-values.
    heights = np.array(heights)
    cents = np.array(cents)
    maxy = np.max(heights)
    miny = np.min(heights)
    decay_pos = np.where(heights == find_nearest(heights, maxy / e))[0][0]
    decay_pos_time = cents[decay_pos]
    avg_y_sep = abs(np.mean(np.diff(heights)))
    efit = Model(echo_as_T2)
    print(efit, echo_as_T2)
    param = efit.make_params()
    param.add('M0', value=maxy, min=maxy * 0.5, max=maxy + (avg_y_sep * 5))
    param.add('T2', value=decay_pos_time, min=decay_pos_time * 0.1, max=decay_pos_time * 2.5)
    param.add('c', value=miny * 0.3, min=miny * 0.1, max=miny * 2.2)
    param.add('ph', value=cents[0] * 0.5, min=0, max=cents[0] * 2)
    param.add('m1', value=0.1)
    param.add('m', value=1)
    result_2 = efit.fit(heights, param, t=cents, method='leastsq', weights=np.mean(np.diff(dat1['T'])) / heights)
    print(result_2.fit_report())
    print('\n', result_2.params.pretty_print())
    ax = plt.gca()
    ax.set_xlabel('Time (us)', fontsize=14)
    ax.set_ylabel('Voltage (V)', fontsize=14)
    xes = np.linspace(np.min(cents), np.max(cents), 100)
    y = efit.eval(t=xes, params=result_2.params)
    plt.plot(xes, y, antialiased=True)
    plt.plot(cents, heights, 'x', ms=8, color='k')
    plt.plot(dat1['V'], dat1['T'], lw=2, antialiased=True,
             color='#4a4a4a', zorder=1)
    plt.title(filename)
    plt.xlim(left=0, right=np.max(cents) * 1.1)
    # plt.ylim(bottom=0, top=result_2.params['M0'].value * 1.1)
    plt.axhline(result_2.params['M0'].value, color='k', ls='--', alpha=0.7, lw=1, zorder=2)
    plt.axhline(result_2.params['M0'].value / e, color='k', ls='--', alpha=0.7, lw=1, zorder=2)
    plt.text(0.9, 0.9, "T_1: {:.4f} us".format(result_2.params['T2'].value), horizontalalignment='center',
             verticalalignment="center",
             transform=ax.transAxes,
             bbox={'pad': 8, 'fc': 'w'}, fontsize=14)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=13)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def fourier_transformer():
    """
    Fourier transforms the combined FID signals of different chemical sites to give a frequency (NMR) spectrum.
    """
    dat, filename = pick_dat(['T', 'V'], 'Sweep_dat', 'Select data to be Fourier Transformed')
    dat['V'] = savgol_filter(dat['V'], 33, 5)
    lower_lim = 300
    upper_lim = 4700
    dat = dat[lower_lim:upper_lim]
    sample_rate = round(1 / np.mean(np.diff(dat['T'])), 11)
    length = len(dat['T'])
    fo = fftpack.fft(dat['V'])
    freq4 = [x * sample_rate / length for x in np.array(range(0, length))]
    halfln = int(length / 2)
    plt.title('{}'.format(filename))
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.plot(dat['T'], dat['V'])
    plt.show()
    fig, ax = plt.subplots()
    plt.title('{} Fourier Transformed'.format(filename))
    figure, = ax.plot(freq4[5:100], abs(fo[5:100]))
    Sel = RangeTool(freq4[5:100], abs(fo[5:100]), figure, ax, 'thing')
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


# initial = [-2.22e-1, 9.31612e-3, 5.85125e-4, 3.609e+2, -1.0822e+1]
# bounds = [25, 0, 0, 0, 1e-6, -1, -50, 252], \
#          [50, 1000, 1000, 1, 1e-4, 0, -1, 452]


def rabi_fit():
    dat, filename = pick_dat(['T', 'V'], 'Sweep_dat', 'Select data to be Fourier Transformed')
    lower_lim = 140
    upper_lim = 5000
    dat = dat[lower_lim:upper_lim]
    initial = [0.012, 1 / 1200, -27, -5, 0, -2, 0.01]
    bounds = [0.009, 1 / 2000, -30, -20, -500, -np.pi, -1], [0.02, 1 / 100, -25, 20, 500, np.pi, 1]
    popt, pcov = curve_fit(Rabi_3, xdata=dat['T'], p0=initial, bounds=bounds, ydata=dat['V'], maxfev=100000,
                           method='trf')
    print(popt)
    plt.plot(dat['T'], dat['V'])
    plt.plot(dat['T'], Rabi_3(dat['T'], *popt))
    fullscreen()
    plt.show()


# rabi_fit()
# fourier_transformer()
pick_ranges()
simple_echo_fits()
# bulk_plot()
# error_prop_sympy()
# bulk_fit()
# read_scan()
# fit_gauss(True, cdf=False, residuals=False)
# vectorized_freq_as_curr_fitting()
# freq_as_curr_fitting()
