from distutils.core import setup, Extension
import os

python_dir = os.listdir('.')
#files_to_skip = ["setup.py", "LICENSE"]
files_to_skip = ["LICENSE"]

for dir in python_dir:
    if dir[0:1] == "." or (dir in files_to_skip) or (not os.path.isdir(dir)):
        continue
    print dir
    os.chdir(dir)
    if os.path.isfile("setup.py"):
        os.system("sudo python setup.py install")
    os.chdir("..")
