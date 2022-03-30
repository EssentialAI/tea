import os
import sys
import shutil

# Get directory name
mydir= r"C:\Users\nares\Downloads\Repos\new repos\EssentialAI.github.io\_build"

try:
    shutil.rmtree(mydir)
    print("Removed")
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))
