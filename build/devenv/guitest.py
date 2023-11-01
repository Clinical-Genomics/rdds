from os import environ

print(environ['DISPLAY'])
#print(environ['XAUTHORITY']) # Must be set to /run/user/1000/gdm/Xauthority

import matplotlib.pyplot as plt
f = plt.plot([1, 2, 3])
plt.show()
