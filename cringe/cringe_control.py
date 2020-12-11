'''
Helper for talking to cringe remotely

G Hilton 2020-12
'''
import zmq

CRINGE_PORT = 5509

CRINGE_COMMANDS = {
    'setup_crate':{'fname':'full_crate_init', 'args':None, 'help':'Full crate init'},
    'full_tune':{'fname': 'extern_tune', 'args':None, 'help':'Full tune'},
    'get_class_var':{'fname':'get_class_var', 'args':['varname'], 'help':'Return the str rep of a class variable'},      
    'get_fb_lock':{'fname':'get_fb_lock', 'args':['column', 'row'], 'help':'Check the state of the fb lock. Requires col, row a/b'},      
    'set_fb_lock':{'fname':'set_fb_lock', 'args':['column', 'row', 'lock'], 'help':'Set the state of the fb lock. Requires state, col, row, a/b'},            
#    'devtest':{'fname':'devtest', 'args':['arg1'], 'help':'temporary for development testing'},      
#    'cmd':{'fname':name, 'args':[], 'help':''},      
    }


def build_zmq_addr(host='localhost', port=CRINGE_PORT):
    return 'tcp://%s:%d' % (host,port)

class CringeControl:
    ''' class for your script to send remote commands to cringe'''
    def __init__(self, host='localhost'):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(build_zmq_addr(host))

    def send(self, message):
        self.socket.send(message.encode('ascii', 'ignore'))
        reply = self.socket.recv().decode()
        return reply

    def setup_crate(self):
        ''' This command was in cringe before I added this module'''
        command = 'setup_crate'
        reply = self.send(command)
        return reply

    def full_tune(self):
        ''' This command was in cringe before I added this module'''
        command = 'full_tune'
        reply = self.send(command)
        return reply

    def get_class_var(self, varname):
        ''' Get the str rep of a class variable of cringe - for example
        load_filename  will get the name of the current config file'''
        command = 'get_class_var'
        if len(varname.split()) != 1:
            return 'bad_argument: no spaces in varname'
        reply = self.send(' '.join((command, varname)))
        return reply

    def get_fb_lock(self, col, row, a_or_b='A' ):
        ''' gets the current fb lock state of mux position'''
        mycol = str(int(col))
        myrow = str(int(row))
        myaorb = None
        if a_or_b.upper() == 'A':
            myaorb = 'A'
        elif a_or_b.upper() == 'B':
            myaorb = 'B'
        if myaorb is None:
            return 'bad_argument: only A or B'
        command = 'get_fb_lock'
        reply = self.send(' '.join((command, mycol, myrow, myaorb)))
        return reply

    def set_fb_lock(self, state, col, row, a_or_b='A' ):
        ''' sets the current fb lock state of mux position'''
        if state:
            mystate = '1'
        else:
            mystate = '0'
        mycol = str(int(col))
        myrow = str(int(row))
        myaorb = None
        if a_or_b.upper() == 'A':
            myaorb = 'A'
        elif a_or_b.upper() == 'B':
            myaorb = 'B'
        if myaorb is None:
            return 'bad_argument: only A or B'
        command = 'set_fb_lock'
        reply = self.send(' '.join((command, mystate, mycol, myrow, myaorb)))
        return reply

    def test(self):
        command = 'devtest'
        reply = self.send(command)
        return reply

def cc_help():
    help_text = []
    for cmd in CRINGE_COMMANDS:
        help_text.append('%s : %s' % (cmd, CRINGE_COMMANDS[cmd]['help']))
    return '\n'.join(help_text)


def cringe_control_commandline_main():
    ''' a commandline client for playing with this'''
    cc = CringeControl()
    print('Talk do cringe - enter a command, quit or help')
    while True:
        request_text = input('?: ')
        if request_text.lower().strip() == 'quit':
            break
        elif request_text.lower().strip() == 'help':
            print(cc_help())
            continue
        request_words = request_text.split()
        if request_words[0].lower() not in  CRINGE_COMMANDS:
            print('Command %s not found' % request_words[0])
            print(cc_help())
            continue
        result = cc.send(request_text)
        print('   Reply: %s' % result)


if __name__ == '__main__':
    cringe_control_commandline_main()