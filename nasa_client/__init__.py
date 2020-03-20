"""A lightweight client for talking to an NDFB Server.

The idea is that this can be the heart of several client programs, including detector relocking,
IV curves, SQUID tuning, curing cancer, and similar activities.  

The key classes are GUIClient (for GUI-based clients) and Client (for non-GUI clients). 
Neither of these are capable of doing much beyond the basic establishing of connections.
The program channel_monitor is a first attempt at using these for good, rather than evil. 

Started September 2011.

Joe Fowler, NIST
"""

__all__ = ['client', 'generic_gui_client']

from client import Client
from generic_gui_client import GUIClient
