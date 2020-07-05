import sys
import subprocess
from subprocess import Popen, PIPE



p = subprocess.Popen(["python", "-W", "ignore", "train.py",
                      #"-d", "cfg/custom_usba.data",
                      #"-c", "cfg/yolov3-custom_usba.cfg",
                      #"-w", "weights/000226.weights"], 
              stdout=sys.stdout)
p.communicate()
