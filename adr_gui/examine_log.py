""" example script to look at adr gui logs"""
# import numpy as np
# import pylab as plt
# import matplotlib.dates as mdates
# import matplotlib.cbook as cbook
# import datetime
# import os

# magup_max = 85

# def load_horton_log(filename):
#    t,temp,heaterout=np.loadtxt(filename,usecols=[1,2,3],delimiter=",",unpack=True, skiprows=1)
#    return t,temp,heaterout

# def find_end_of_demags(t,temp,heaterout):
#    inds = np.where(np.diff(np.array(heaterout>magup_max,dtype="int"))==-1)[0]
#    return t[inds], temp[inds], heaterout[inds]

# def get_log_filenames(dirpath):
#    potentials = os.listdir(dirpath)
#    return [p for p in potentials if os.path.isfile(p) and "hortonlog" in p]

# def get_postmag_statuses(dirpath):
#    filenames = get_log_filenames(dirpath)
#    t_, temp_, heaterout_ = [],[],[]
#    for filename in filenames:
#       try:
#          t,temp,heaterout = load_horton_log(filename)
#       except ValueError:
#          continue # avoid "need more than 0 values to unpack" 
#       if isinstance(t,float): continue # skip files with only one entry
#       # they wont have interesting data, and they return a different datatype
#       # which breaks following code
#       t__, temp__, heaterout__ = find_end_of_demags(t,temp,heaterout)
#       t_ += list(t__)
#       temp_ += list(temp__)
#       heaterout_ += list(heaterout__)
#    return t_, temp_, heaterout_

# def co_sort_by_time(t,temp,heaterout):
#    inds = np.argsort(t)
#    return np.array(t)[inds], np.array(temp)[inds], np.array(heaterout)[inds]

# t,temp,heaterout = get_postmag_statuses(os.path.expanduser( "~/ADRLogs"))
# td = np.array([datetime.datetime.fromtimestamp(x) for x in t])

# fig, ax = plt.subplots()
# ax.plot(td,temp,"o")
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
# plt.xlabel("date")
# plt.ylabel("2nd stage PT temperature (K)")


import numpy as np
import pylab as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import datetime
import os
filename = "ADRLog_20210728_t112958.txt"
filename = os.path.join(os.path.expanduser( "~/ADRLogs"), filename)

t,temp,heaterout=np.genfromtxt(filename, delimiter=",", 
usecols=[1,2,3], invalid_raise=False, unpack=True)

plt.ion()
plt.figure()
plt.plot((t-t[0])/3600,temp)
plt.xlabel("time (hours)")
plt.ylabel("temp (K)")

plt.figure()
plt.plot((t-t[0])/3600,heaterout)
plt.xlabel("time (hours)")
plt.ylabel("heater out (%)")