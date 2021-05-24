import nasa_client
import pylab as plt
import numpy as np

plt.ion()
# plt.close("all")

c = nasa_client.EasyClient()
N=10000
data = c.getNewData(minimumNumPoints=N, exactNumPoints=True)


plt.figure()
plt.plot(data[1,1,::2,1], label="ivdata")
plt.plot(data[0,2,::2,1], label="triangle")
plt.xlabel("point num")
plt.ylabel("iv data")
plt.legend("20210507_SSRL_AX_56p6_DFB_IV.npy")

tri = data[0,2,:,1]
fb = data[1,1,:,1]

# np.save("20210507_SSRL_AX_56p6_DFB_IV.npy", data)
# plt.figure()
# plt.plot(data[0,0,:,1]+np.arange(N)/100,data[1,0,:,1])
# plt.xlabel("triangle")
# plt.ylabel("iv fb")


# for i in range(0,20,2):
#     plt.plot(tri[i::2**10], fb[i::2**10], label=f"{i}")
# plt.legend()