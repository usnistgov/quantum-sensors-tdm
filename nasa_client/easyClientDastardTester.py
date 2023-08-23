from nasa_client import easyClientDastard

c = easyClientDastard.EasyClientDastard()
c.setupAndChooseChannels()
data = c.getNewData()
data2 = c.getNewData2(4000)
print(data)