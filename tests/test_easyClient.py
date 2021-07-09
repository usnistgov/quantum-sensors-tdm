''' 
test_easyClient.py 
'''
from nasa_client import EasyClient
import matplotlib.pyplot as plt


def test_easyClient():
    ec = EasyClient()
    ec.setupAndChooseChannels()
    #format of data returned is dataOut[col,row,frame,error=0/fb=1]
    dataOut = ec.getNewData(delaySeconds = 0.001, minimumNumPoints = 4000, exactNumPoints = False, sendMode = 0, toVolts=False, divideNsamp=True, retries = 3)
    
    for ii_col in range(ec.numColumns):
        fig,ax = plt.subplots(2,num=ii_col)
        for jj_row in range(ec.numRows):
            ax[0].plot(dataOut[ii_col,jj_row,:,0],label=jj_row)
            ax[1].plot(dataOut[ii_col,jj_row,:,1])
        ax[0].set_title('Error')
        ax[1].set_title('Feedback')
        ax[0].legend()
        fig.suptitle('Column: %d'%ii_col)
    plt.show()

if __name__ == "__main__":
    test_easyClient()
    
    

