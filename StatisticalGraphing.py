from tkinter import filedialog, Tk

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel
from pandas import read_csv, DataFrame
from uncertainties import ufloat

from ResidualPlotter import MultiPlot
from range_selector import RangeTool

plt.switch_backend('QT5Agg')


def fullscreen():
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()


def residuals(force_pickrange=False):
    datafolder = filedialog.askopenfilenames(initialdir="C:\\Users\Josh\IdeaProjects\OpticalPumping",
                                             title="Select data for bulk plotting")
    for filename in datafolder:
        if 'data' in filename:
            global dat3
            name_ext = filename.split('/')[-1]
            name_no_ending = name_ext.split('.csv')[0]
            parts = name_no_ending.split('_')
            print(parts)

            dat1 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_dat\\{}".format(name_ext),
                            names=['xdat', 'ydat'])
            dat1 = dat1[np.abs(dat1['xdat'] - dat1['xdat'].mean()) <= (3 * dat1['xdat'].std())]
            xdat = np.array(dat1['xdat'])
            ydat = np.array(dat1['ydat'])
            if force_pickrange:
                fig1, ax1 = plt.subplots()
                plt.title('Pick Ranges for Exponential decay fit')
                Figure1, = ax1.plot(xdat, ydat, '.')
                print(xdat[5], xdat[-5])
                plt.xlim([xdat[5], xdat[-5]])
                Sel3 = RangeTool(xdat, ydat, Figure1, ax1, name_no_ending)
                fullscreen()
                plt.show()
                dat3 = Sel3.return_range()
            if not force_pickrange:
                try:
                    dat3 = read_csv("C:\\Users\\Josh\\IdeaProjects\\OpticalPumping\\Sweep_ranges\\{}".format(name_ext),
                                    names=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
                except FileNotFoundError or force_pickrange:
                    fig1, ax1 = plt.subplots()
                    plt.title('Pick Ranges for Exponential decay fit')
                    Figure1, = ax1.plot(xdat, ydat, '.')
                    print(xdat[5], xdat[-5])
                    plt.xlim([xdat[5], xdat[-5]])
                    Sel3 = RangeTool(xdat, ydat, Figure1, ax1, name_no_ending)
                    fullscreen()
                    plt.show()
                    dat3 = Sel3.return_range()

            mdl = LorentzianModel()
            try:
                lowerindex = int(dat3.at[0, 'LowerIndex'])
                upperindex = int(dat3.at[0, 'UpperIndex'])
            except ValueError:
                pass
            try:
                params = mdl.guess(data=ydat[lowerindex:upperindex], x=xdat[lowerindex:upperindex])
                result = mdl.fit(ydat[lowerindex:upperindex], params, x=xdat[lowerindex:upperindex])
                resultdata = mdl.eval(x=xdat[lowerindex:upperindex], params=result.params)
                # print(result.fit_report())
                MultiPlot(xdat[lowerindex:upperindex], ydat[lowerindex:upperindex], resultdata, xdat, ydat,
                          name_no_ending, parts[1])
            except UnboundLocalError:
                pass


residuals(force_pickrange=False)
