from detchar import IVCurveColumnData, IVTempSweepData, IVColdloadSweepData
from . import test_data
import matplotlib.pyplot as plt
import time, os

def pause_ci_safe(t):
    v = os.environ.get("CI", False)
    if not v:
        tstart = time.time()
        while time.time()-tstart < t:
            # small pauses until total duration has elapsed
            # use plt.pause to allow plots to appear
            plt.pause(5)
            any_figs_open = len(plt.get_fignums()) > 0
            if not any_figs_open:
                break

# def test_readwrite_IVCurveColumnData():
#     data = IVCurveColumnData.from_file(test_data.ivtest_filename)
#     plt.ion()
#     data.plot()
#     pause_ci_safe(.5)

#     data2 = IVCurveColumnData.from_file(test_data.ivtest_filename)
#     data2.to_file("test_data/IVCurveColumnData_writetest.json", overwrite=True)

#     data3 = IVCurveColumnData.from_file("test_data/IVCurveColumnData_writetest.json")
#     data3.plot()
#     pause_ci_safe(.5)

# def test_readwrite_IVTempSweepData():
#     data = IVCurveColumnData.from_file(test_data.ivtest_filename)
#     data2 = IVCurveColumnData.from_file(test_data.ivtest_filename)
#     #data2.fb_values = data2.fb_values.array()+1
#     sweepdata = IVTempSweepData([.1,.2],[data,data2])
#     sweepdata.to_file('test_data/ivtempsweep_test_data.json',True)

#     sweepdata2 = IVTempSweepData.from_file('test_data/ivtempsweep_test_data.json')
#     sweepdata2.plot_row(0)
#     pause_ci_safe(.5)
    
# def test_readwrite_IVTempSweepDataGalen():
#     sweepdata = IVTempSweepData.from_file('test_data/iv_sweep_test_galen.json')
#     sweepdata.plot_row(0)
#     pause_ci_safe(.5)

# def test_load_IVsweep():
#     plt.ion()
#     data = IVTempSweepData.from_file('test_data/lbird_hftv0_ivsweep_test.json')
#     for ii in range(24):
#         data.plot_row(ii)
#         pause_ci_safe(1)

# def test_coldload_io():
#     plt.ion()
#     data = IVColdloadSweepData.from_file('test_data/coldload_sweep_test.json')
#     for ii in range(24):
#         data.plot_row(ii)
#         pause_ci_safe(1)

def test_iv_plot():
    plt.ion()
    data = IVCurveColumnData.from_file('test_data/lbird_iv_normalbranch_20210130.json')
    print(data.nominal_temp_k)
    data.plot()

    pause_ci_safe(10)

    
