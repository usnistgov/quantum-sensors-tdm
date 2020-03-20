import yaml
from os import path
import os
import time

trigger_rate_filename = path.expanduser("~/.daq/trigger_rates.yaml")

def load_trigger_rates():
    with open(trigger_rate_filename) as f:
        triggerdict = yaml.load(f)
    return triggerdict

def total_edge_trigger_rate(loadfunc = load_trigger_rates):
    triggerdict = loadfunc()
    if triggerdict["trigger-conditions"]["edge-trig"]==True and triggerdict["trigger-conditions"]["auto-trig"] == False:
        return triggerdict["total-rate"]
    else: 
        return 0.0
    
def load_fresh_rates():
    triggerdict = load_trigger_rates()
    integration_time = triggerdict["rate-integration-time"]
    last_file_time = os.stat(trigger_rate_filename)[-1]
    while os.stat(trigger_rate_filename)[-1]-last_file_time <= integration_time:
        time.sleep(integration_time-os.stat(trigger_rate_filename)[-1]+last_file_time)
    return load_trigger_rates()

def freshrate():
    return total_edge_trigger_rate(load_fresh_rates)