import nasa_client
import pylab as plt
import numpy as np

plt.ion()
# plt.close("all")

c = nasa_client.EasyClient()
N=2**20
data = c.getNewData(minimumNumPoints=N, exactNumPoints=True)

tri = data[1,0,:,1]
fb = data[0,0,:,1]

plt.figure()
plt.plot(fb, label="ivdata")
plt.plot(tri, label="triangle")
plt.xlabel("point num")
plt.ylabel("iv data")
plt.legend()



np.save("latest.npy", data)
# plt.figure()
# plt.plot(data[0,0,:,1]+np.arange(N)/100,data[1,0,:,1])
# plt.xlabel("triangle")
# plt.ylabel("iv fb")


# for i in range(0,20,2):
#     plt.plot(tri[i::2**10], fb[i::2**10], label=f"{i}")
# plt.legend()