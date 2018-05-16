import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from numpy import sqrt, asarray, var, linspace
from seaborn import set_style

set_style("whitegrid")


def MultiPlot(xdata, ydata, fitdata, fullx, fully, key, title):
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.rcParams['text.latex.preamble'] = [
    #     r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
    #     r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
    #     r'\usepackage{lmodern}',  # set the normal font here
    #     ]
    # plt.rcParams['ps.usedistiller'] = 'xpdf'
    # plt.rcParams['ps.distiller.res'] = '1600'
    labelsize = 14
    ticksize = 12
    fig = plt.figure(figsize=(8, 5.5))
    # fig.subplots_adjust(hspace=0, wspace=0)
    ax1 = fig.add_subplot(2, 2, 1)
    fig.subplots_adjust(top=0.932,
                        bottom=0.106,
                        left=0.118,
                        right=0.974,
                        hspace=0.307,
                        wspace=0.262)
    ax1.plot(xdata / 1000, fitdata, antialiased=True, label='Lorentzian fit')
    ax1.plot(fullx / 1000, fully, '.', color='#1c1c1c', label='Sweep data', zorder=0, ms=1.5)
    dely = sqrt(sqrt(ydata) ** 2 + 0.5 ** 2)
    residuals = ydata - fitdata
    ax1.fill_between(xdata / 1000, fitdata - dely,
                     fitdata + dely, color="#ABABAB", label=r'$1\sigma$', alpha=0.7)
    ax1.grid(color='k', linestyle='--', alpha=0.2)
    plt.xlabel('Frequency (KHz)', fontsize=labelsize)
    plt.ylabel('Variance (a.u.)', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    ax1.legend(frameon=True, loc='best')
    # plt.title('Peak with 1 sigma error bands')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(xdata / 1000, residuals, '.', antialiased=True, ms=1.5, color='#1c1c1c')
    ax2.grid(color='k', linestyle='--', alpha=0.2)
    plt.xlabel('Frequency (KHz)', fontsize=labelsize)
    plt.ylabel(r'$y_m - y_f$', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    # plt.title('Residuals')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(xdata / 1000,
             (residuals ** 2) / dely ** 2,
             '.', antialiased=True, ms=1.5, color='#1c1c1c')
    ax3.grid(color='k', linestyle='--', alpha=0.2)
    plt.xlabel('Frequency (KHz)', fontsize=labelsize)
    plt.ylabel(r'$\frac{y_m - y_f}{\sigma_y}$', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    # plt.title('Normalised residuals')

    ax4 = fig.add_subplot(2, 2, 4)
    n, bins, patches = ax4.hist(residuals, bins='auto', label='Residual hist.', color='#1c1c1c', alpha=0.9)

    mdl = GaussianModel()
    bin_centre = []
    for t in range(0, len(bins) - 1):
        bin_centre.append((bins[t + 1] + bins[t]) / 2)
    bin_centre2 = asarray(bin_centre)
    pars = mdl.guess(n, x=bin_centre2)
    result2 = mdl.fit(n, pars, x=bin_centre2)
    xs = linspace(min(bin_centre), max(bin_centre), 1000)
    ys = mdl.eval(x=xs, params=result2.params)
    corr_coeff = 1 - result2.residual.var() / var(n)
    # at = AnchoredText("$R^2 = {:.3f}$".format(corr_coeff),
    #                   prop=dict(size=10), frameon=True,
    #                   loc=2,
    #                   )
    # ax4.add_artist(at)
    ax4.plot(0, 0, '.', color='k', ms=0, label="$R^2 = {:.3f}$".format(corr_coeff))
    ax4.plot(xs, ys, antialiased=True, label='Lorentzian fit')
    ax4.grid(color='k', linestyle='--', alpha=0.2)
    ax4.legend(frameon=True, loc='best')
    plt.xlabel(r'$y_m - y_f$', fontsize=labelsize)
    plt.ylabel('Counts', fontsize=labelsize)
    plt.tick_params(axis='both', which='major', labelsize=ticksize)
    # plt.title('Residual histogram')
    fig.suptitle('Helmholtz coil current: {}'.format(title))
    # fig.tight_layout()
    # fig.set_size_inches(16.5, 10.5)
    # fig_manager = plt.get_current_fig_manager()
    # fig_manager.window.showMaximized()

    plt.savefig("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\MatplotlibFigures\\Lorentzian_{}.png".format(
            key), dpi=600)
    # plt.show()
    plt.close()
