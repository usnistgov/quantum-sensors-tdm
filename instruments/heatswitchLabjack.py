import labjack
import time

class HeatswitchLabjack():
    
    
    def __init__(self):
        self.lj = labjack.Labjack()
        
        
        # prepare for use
        self.outputChannels = {'open':17,'close':16, 'notUsed1':13, 'notUsed2':14, 'passThru12':9, 'passThru13':11 }
        self.inputChannels = {'closing':19, '4K1KTouch':4, '4K50mKTouch':12, '1K50mKTouch':14, 
                         'anyTouch':18}
        
        for channelName in self.outputChannels:
            self.lj.setDigIOState(self.outputChannels[channelName], 'low')
            self.lj.setDigIODirection(self.outputChannels[channelName], 'output' )
        for channelName in self.inputChannels:
            self.lj.setDigIODirection(self.inputChannels[channelName], 'input')
#        print('Set labjack channels for heatswitch use\n')    
            
            
    def readTouchChecks(self):
        touch = {}
        touch['anyTouch'] = self.lj.getDigIOState(self.inputChannels['anyTouch'])
        touch['4K1KTouch'] = self.lj.getDigIOState(self.inputChannels['4K1KTouch'])
        touch['4K50mKTouch'] = self.lj.getDigIOState(self.inputChannels['4K50mKTouch'])
        touch['1K50mKTouch'] = self.lj.getDigIOState(self.inputChannels['1K50mKTouch'])
        
        return touch
    
    def readWorkingTouchCheck(self):
        
        if self.lj.getDigIOState(self.inputChannels['4K50mKTouch']) == 'low':
            return True
        else: return False
    
    def readClosing(self):
        closing = self.lj.getDigIOState(self.inputChannels['closing'])
        return closing
        
    def sendOpen(self):
        "sets open chanel high then low to active heatswitch controller"
        self.lj.setDigIOState(self.outputChannels['open'],'high')
        time.sleep(0.01)
        self.lj.setDigIOState(self.outputChannels['open'],'low')
        time.sleep(5)
        if self.readWorkingTouchCheck() is True:
            print("warning: still see touch after Heatswitch open, this is normal if magged up")
            return False
        else: 
            print('Heatswitch open looks succesful, I see no electrical touch')
            return True
    
    def sendClose(self):
        "sets close chanel high then low to active heatswitch controller"
        self.lj.setDigIOState(self.outputChannels['close'],'high')
        time.sleep(0.01)
        self.lj.setDigIOState(self.outputChannels['close'],'low')
        time.sleep(5)
        if self.readWorkingTouchCheck() is False:
            print("WARNING: heat switch does not seem to have closed, trying open then 2x close again")
            self.sendOpen()
            self.lj.setDigIOState(self.outputChannels['close'],'high')
            time.sleep(0.01)
            self.lj.setDigIOState(self.outputChannels['close'],'low')
            time.sleep(5)
            self.lj.setDigIOState(self.outputChannels['close'],'high')
            time.sleep(0.01)
            self.lj.setDigIOState(self.outputChannels['close'],'low')
            time.sleep(5)
            print("done trying open and 2x close")
            if self.readWorkingTouchCheck() is False:
                print("WARNING: don't see touch after Heatswitch close, this is a problem!! I already tried sending an open and 2x close again")
                return False
            else: 
                print('Heatswitch close looks succcesful, I see an electrical touch.')
                return True
        else: 
            print('Heatswitch close looks succcesful, I see an electrical touch.')
            return True
    def OpenHeatSwitch(self):
        
        self.sendOpen()
        
    def CloseHeatSwitch(self):
        self.sendClose()