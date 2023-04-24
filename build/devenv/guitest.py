from os import environ
import matplotlib as mpl
mpl.use('tkagg')

print(environ['DISPLAY'])
#print(environ['XAUTHORITY']) # Must be set to /run/user/1000/gdm/Xauthority

import matplotlib.pyplot as plt
f = plt.plot([1, 2, 3])
plt.show()
