import shutil
import os

source = './inputs/'
dest1 = './moved_inputs'

files = os.listdir(source)

for f in files:
    # print(f)
    if "100.in" in f or "200.in" in f:
        print(f)
        shutil.move(source+f, dest1)