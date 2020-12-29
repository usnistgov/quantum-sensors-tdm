'''
Helper for talking to adr_gui remotely

Based on cringe_control.py by G Hilton 2020-12
'''
import zmq

ADR_GUI_PORT = 5020

ADR_GUI_COMMANDS = {
    'get_temp_k': {'fname':'rpc_get_temp_k', 'args':None, 'help':'return the most recent FAA temp reading in kelvin'},
    'set_temp_k': {'fname':'rpc_set_temp_k', 'args':['setpoint_k'], 'help':'works in control mode only, set the control loop temperature setpoint in kelvin'},
    'get_temp_rms_uk': {'fname':'rpc_get_temp_rms_uk', 'args':None, 'help':'temp temperature stability rms'},
    'get_slope_hout_per_hour': {'fname':'rpc_get_slope_hout_per_hour', 'args':None, 'help':'return the heater change per hour slope'},
    'get_hout': {'fname':'rpc_get_hout', 'args':None, 'help':'return the heater out value'},
    'echo':{'fname':'rpc_echo', 'args':['x'], 'help':'return the string representation of x, for testing'},      
    }


def build_zmq_addr(host='localhost', port=ADR_GUI_PORT):
    return f'tcp://{host}:{port}'

class AdrGuiControl:
    ''' class for your script to send remote commands to adr_gui'''
    def __init__(self, host='localhost'):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(build_zmq_addr(host))

    def send(self, message):
        self.socket.send(message.encode('ascii', 'ignore'))
        reply = self.socket.recv().decode()
        return reply

    def send_decode_reply(self, message):
        reply = self.send(message)
        a, b = reply.split(":", 1)
        success = a == "ok"
        return success, b

    def echo(self, x):
        x=str(x)
        command = 'echo'
        success, extra_info = self.send_decode_reply(f"{command} {x}")
        assert success
        assert extra_info == x
        return reply

    def get_temp_k(self):
        success, extra_info = self.send_decode_reply("get_temp_k")
        assert success
        return float(extra_info)

    def set_temp_k(self, setpoint_k):
        success, extra_info = self.send_decode_reply(f"get_temp_k {float(setpoint_k)}")
        return success

    def get_temp_rms_uk(self):
        success, extra_info = self.send_decode_reply("get_temp_rms_uk")
        assert success
        return float(extra_info)   

    def get_hout(self):
        success, extra_info = self.send_decode_reply("get_hout")
        assert success
        return float(extra_info)   

    def get_slope_hout_per_hour(self):
        success, extra_info = self.send_decode_reply("get_slope_hout_per_hour")
        assert success
        return float(extra_info)   

def cc_help():
    help_text = []
    for cmd in ADR_GUI_COMMANDS:
        help_text.append('%s : %s' % (cmd, ADR_GUI_COMMANDS[cmd]['help']))
    return '\n'.join(help_text)


def adr_gui_commandline_main():
    ''' a commandline client for playing with this'''
    cc = AdrGuiControl()
    print('Talk do adr_gui - enter a command, quit or help')
    while True:
        request_text = input('?: ')
        if request_text.lower().strip() == 'quit':
            break
        elif request_text.lower().strip() == 'help':
            print(cc_help())
            continue
        request_words = request_text.split()
        if len(request_words) == 0:
            print(cc_help())
            continue
        if request_words[0].lower() not in ADR_GUI_COMMANDS:
            print('Command %s not found' % request_words[0])
            print(cc_help())
            continue
        result = cc.send(request_text)
        print('   Reply: %s' % result)


if __name__ == '__main__':
    adr_gui_commandline_main()