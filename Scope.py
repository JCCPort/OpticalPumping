from time import strftime, gmtime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvisa
from scipy.signal import savgol_filter

start = time()

today = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
plt.switch_backend('QT5Agg')

# pyvisa.log_to_screen()
rm = pyvisa.ResourceManager('@ni')
# print(rm.list_resources())
inst = rm.open_resource("USB0::0x0699::0x0363::C059771::INSTR", encoding='utf8')
inst2 = rm.open_resource("USB0::0x0699::0x03B0::C010710::INSTR", encoding='utf8')
inst.read_termination = '\n'
inst.write_termination = '\n'
inst2.read_termination = '\n'
inst2.write_termination = '\n'
inst.write("DATA:SOURCE CH1")
inst2.write("DATA:SOURCE CH1")
# print(inst2.session)
query = inst2.query
query_binary = inst.query_binary_values
arr = np.array
vari = np.var


def f():
    try:
        freq = float(query(message='TRIGger:MAIn:FREQuency?'))
        rawdat_transdat = arr(query_binary(message='CURV?', datatype='b',
                                           is_big_endian=True))
        xarr = np.linspace(0, 4999, 5000)
        # plt.plot(rawdat_transdat)
        # plt.show()
        var = vari(savgol_filter(rawdat_transdat, 113, 1))
        # print(var)
        return var, freq, rawdat_transdat, xarr
    except pyvisa.errors.VisaIOError:
        pass


# vs = []
# fs = []
# k = 1
# elapsed = 0
# data = pd.DataFrame(columns=['f', 'H'])
# while elapsed < (13 * 60):
#     elapsed = time() - start
#     print(elapsed)
#     vars, freqs, rawdat_transdat = f()
#     data.at[k, 'H'] = vars
#     data.at[k, 'f'] = freqs
#     k += 1


vars, freqs, rawdat, xs = f()
print(rawdat)
data = pd.DataFrame(np.transpose([xs, savgol_filter(rawdat, 133, 3)]), columns=['T', 'V'])
# print('Iterations per second:   {}'.format(k / elapsed))
current = input('What is the current?')
VPP = input('What is the VPP Amplitude?')
#
# print(vs, fs)
# data = data.drop(data[data['f'] > 1e7].index)

fig, ax = plt.subplots()
# data.sort_values(['f'], inplace=True)
data.to_csv('C:\\Users\Josh\IdeaProjects\OpticalPumping\Sweep_dat\Rabi_{}__{:.2f}Hz__{}VPP.csv'.format(today, freqs,
                                                                                                       VPP,
                                                                                                       current),
            index=False, header=False)
# plt.plot(data['T'], medfilt(data['V'], 5), '.')
plt.plot(data['T'], data['V'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Transmission intensity (a.u)')
fig_manager = plt.get_current_fig_manager()
fig_manager.window.showMaximized()
plt.show()
