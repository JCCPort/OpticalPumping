from time import strftime, gmtime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvisa
from scipy.signal import medfilt, savgol_filter

start = time()

today = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
plt.switch_backend('QT5Agg')

# pyvisa.log_to_screen()
rm = pyvisa.ResourceManager('@ni')
# print(rm.list_resources())
inst = rm.open_resource("USB0::0x0699::0x0363::C059771::INSTR", encoding='utf8')
inst2 = rm.open_resource("USB0::0x0699::0x03B0::C010710::INSTR")
inst.read_termination = '\n'
inst.write_termination = '\n'
inst2.read_termination = '\n'
inst2.write_termination = '\n'
inst.write("DATA:SOURCE CH1")
inst2.write("DATA:SOURCE CH1")
print(inst2.session)
query = inst2.query
query_binary = inst.query_binary_values
arr = np.array
vari = np.var


def f():
    try:
        freq = float(query(message='TRIGger:MAIn:FREQuency?'))
        rawdat_transdat = savgol_filter(arr(query_binary(message='CURV?', datatype='b',
                                                         is_big_endian=True)), 113, 1)
        var = vari(rawdat_transdat)
        return var, freq
    except pyvisa.errors.VisaIOError:
        pass


vs = []
fs = []
k = 1
elapsed = 0
data = pd.DataFrame(columns=['f', 'H'])
while elapsed < (5 * 60):
    elapsed = time() - start
    print(elapsed)
    vars, freqs = f()
    data.at[k, 'H'] = vars
    data.at[k, 'f'] = freqs
    k += 1

print('Iterations per second:   {}'.format(k / elapsed))
current = input('What is the current?')

print(vs, fs)
data = data.drop(data[data['f'] > 1e7].index)

fig, ax = plt.subplots()
data.sort_values(['f'], inplace=True)
data.to_csv('C:\\Users\Josh\IdeaProjects\OpticalPumping\Sweep_dat\data_{}A_{}.csv'.format(current, today),
            index=False, header=False)
plt.plot(data['f'], medfilt(data['H'], 5), '.')
plt.plot(data['f'], medfilt(data['H'], 5))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Transmission intensity (a.u)')
fig_manager = plt.get_current_fig_manager()
fig_manager.window.showMaximized()
plt.show()
