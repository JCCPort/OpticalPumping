from time import gmtime, strftime
from tkinter import filedialog, Tk

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import GaussianModel, LinearModel
from pandas import read_csv, read_hdf, DataFrame
from scipy.constants import mu_0, physical_constants, h
from scipy.odr import Model, RealData, ODR
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from seaborn import set_style
from uncertainties import ufloat

from range_selector import RangeTool

today = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
set_style("whitegrid")
# set_palette("Set1")
plt.switch_backend('QT5Agg')
mu_B = physical_constants['Bohr magneton']
electron_g = physical_constants['electron g factor']
m_e = physical_constants['electron mass']
print(m_e)
m_cs = [2.2069468e-25, 3.3210778e-34]
g_I = [-0.00039885395, 0.0000000000052]
lab_field_NOAA = [48585.7 * 0.00001, 152 * 0.00001]
formatted_NOAA = ufloat(lab_field_NOAA[0], lab_field_NOAA[1])


def electron_lande(J, S, L, gL=1 - (m_e[0] / m_cs[0]), gS=electron_g[0]):
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


def hyperfine_lande(F, I, J, gJ, gI=g_I[0]):
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


def hyperfine_lande_uncert(F, I, J, S, L):
    term1 = (((J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1))) * (m_e[2] / m_cs[0])) ** 2
    term2 = (((J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1))) * (m_cs[1] * m_e[0] / m_cs[0] ** 2)) ** 2
    term3 = (((J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))) * electron_g[2]) ** 2
    term4 = (((F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))) * g_I[1]) ** 2
    return np.sqrt(term1 + term2 + term3 + term4)


gF3 = [hyperfine_lande(F=3, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)),
       hyperfine_lande_uncert(F=3, I=7 / 2, J=1 / 2, S=1 / 2, L=0)]
print(hyperfine_lande_uncert(F=3, I=7 / 2, J=1 / 2, S=1 / 2, L=0))
gF2 = [hyperfine_lande(F=2, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)),
       hyperfine_lande_uncert(F=2, I=7 / 2, J=1 / 2, S=1 / 2, L=0)]
formatted_gf3 = ufloat(gF3[0], gF3[1])
formatted_gf2 = ufloat(gF2[0], gF3[1])

print(hyperfine_lande(F=4, I=7 / 2, J=1 / 2, gJ=electron_lande(J=1 / 2, S=1 / 2, L=0)))


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
    dat1, filename1 = pick_dat(['t', 'm'], "Sweep_dat", "Select dataset to draw from")
    dat2 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_ranges\\{}".format(filename1),
                    names=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
    if smooth:
        win_len = int(len(dat1) / 10)
        if win_len % 2 == 0:
            win_len += 1
        dat1['m'] = savgol_filter(dat1['m'], win_len, 1)
    xrange = []
    yrange = []
    xranges = {}
    yranges = {}
    x_append = xrange.append
    y_append = yrange.append
    for o in range(0, len(dat2)):
        x_append((dat1['t'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
        y_append((dat1['m'][dat2['LowerIndex'][o]:dat2['UpperIndex'][o] + 1]).values)
    for o in range(0, len(xrange)):
        xranges[o] = xrange[o]
        yranges[o] = yrange[o]
    return xranges, yranges, xrange, yrange, filename1, dat1


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
    lande_overlap = (abs(popu[0].n - formatted_gf3.n) / (popu[0].std_dev + formatted_gf3.std_dev))
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
    print('\tCalculated Landé g-factor: \t \t \t \t {:.4f}'.format(formatted_gf3))
    print('\tStd. Dev. Separation: \t \t \t \t \t {:.2f}'.format(lande_overlap))
    print('\n')
    print('\tReduced chi-sq: \t \t \t \t \t \t {:.2f}'.format(myoutput.res_var / (len(data['I']) - len(opt_vals))))
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
    ax.plot(plot_vals, vectorized_freq_as_curr(I=plot_vals, P=myoutput.beta), antialiased=True, lw=2.5, label='ODR Fit',
            color='k')
    Sel = RangeTool(data['I'], data['f'], figure2, ax, 'Thing')
    plt.xlim((-max_x * 1.15, max_x * 1.15))
    plt.ylim((0, max_y * 1.15))
    fullscreen()
    plt.legend()
    plt.savefig("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\VectorizedFreqAsCurr_{}.png".format(
            today), dpi=600)
    plt.show()


def read_scan():
    data, filename = pick_dat(['f', 'RT'], 'Sweep_dat')
    fig, ax = plt.subplots()
    figure1, = ax.plot(data['f'], data['RT'], '.', markerfacecolor="None", color='#050505',
                       mew=1.4, ms=1, antialiased=True, label='Data')
    window = int(len(data) / 20)
    if window % 2 == 0:
        window += 1
    figure2, = ax.plot(data['f'], savgol_filter(data['RT'], window, 2), lw=2)
    if filename.endswith('.csv'):
        name = filename[:-4]
    Thing = RangeTool(data['f'], data['RT'], figure1, ax, name)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Variance (a.u)', fontsize=14)
    plt.xlim([np.min(data['f']), np.max(data['f'])])
    ax.axes.tick_params(labelsize=12)
    plt.title('Relative light transmission for a given Helmholtz current as a function of frequency')
    fullscreen()
    plt.savefig("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\Sweep_plot_{}.pdf".format(
            today), dpi=600)
    plt.show()


def fit_gauss(graph=False):
    xranges, yranges, xrange, yrange, filename1, dat1 = range_to_list()
    FitVals = DataFrame(columns=['Sigma', 'Center', 'Amplitude', 'FWHM', 'Height', 'Intercept', 'Slope', 'ChiSq',
                                 'RedChiSq', 'Akaike', 'Bayesian'])
    for i in range(0, len(xranges)):
        mdl = GaussianModel()
        line = LinearModel()
        params = mdl.guess(data=yranges[i], x=xranges[i])
        params += line.guess(data=yranges[i], x=xranges[i])
        model = mdl + line
        result = model.fit(yranges[i], params, x=xranges[i])
        print(result.fit_report())
        FitVals.at[i, 'Sigma'] = ufloat(result.params['sigma'].value, result.params['sigma'].stderr)
        FitVals.at[i, 'Center'] = ufloat(result.params['center'].value, result.params['center'].stderr)
        FitVals.at[i, 'Amplitude'] = ufloat(result.params['amplitude'].value, result.params['amplitude'].stderr)
        FitVals.at[i, 'FWHM'] = ufloat(result.params['fwhm'].value, result.params['fwhm'].stderr)
        FitVals.at[i, 'Height'] = ufloat(result.params['height'].value, result.params['height'].stderr)
        FitVals.at[i, 'Intercept'] = ufloat(result.params['intercept'].value, result.params['intercept'].stderr)
        FitVals.at[i, 'Slope'] = ufloat(result.params['slope'].value, result.params['slope'].stderr)
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


# bulk_fit()
read_scan()
fit_gauss(True)
# vectorized_freq_as_curr_fitting()
# freq_as_curr_fitting()
