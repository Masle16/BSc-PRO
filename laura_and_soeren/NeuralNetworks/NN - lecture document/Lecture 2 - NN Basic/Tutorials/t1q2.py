
import matplotlib.pyplot as plt
import numpy as np

import os
if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

a = np.arange(-1.5, 1.5, 0.05)

# decision boundaries
b1 = -a - 0.5
b2 = 0.5*a + 0.5
b3 = 4*a +1


plt.figure(1)
l1 = plt.subplot(1, 1, 1)
l1.plot(a, b1, color = 'black', linestyle = '--')
l1.fill_between(a, b1, 1.5, facecolor='yellow')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-1.5, 1.5)
plt.xlim(a.min(), a.max())
plt.text(1.3, 0.1, r'$\alpha$')
plt.text(0.1, 1.3, r'$\beta$')
plt.text(0.5, 0.5, 'ON')
plt.text(-1.0, -1.0, 'OFF')
plt.savefig('./figures/figure_t1q2_1.png')


plt.figure(2)
l2 = plt.subplot(1, 1, 1)
l2.plot(a, b2, color = 'black', linestyle = '--')
l2.fill_between(a, b2, -1.5, facecolor='green')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-1.5, 1.5)
plt.xlim(a.min(), a.max())
plt.text(1.3, 0.1, r'$\alpha$')
plt.text(0.1, 1.3, r'$\beta$')
plt.text(0.5, -0.5, 'ON')
plt.text(-1.0, 1.0, 'OFF')
plt.savefig('./figures/figure_t1q2_2.png')

plt.figure(3)
l3 = plt.subplot(1, 1, 1)
l3.plot(a, b3, color = 'black', linestyle = '-')
l3.fill_between(a, 1.5, b3, facecolor='blue')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-1.5, 1.5)
plt.xlim(a.min(), a.max())
plt.text(1.3, 0.1, r'$\alpha$')
plt.text(0.1, 1.3, r'$\beta$')
plt.text(0.5, 0.5, 'ON')
plt.text(-1.0, 1.0, 'OFF')
plt.savefig('./figures/figure_t1q2_3.png')

plt.figure(4)
line = plt.subplot(1, 1, 1)
line.plot(a, b1, color = 'black', linestyle = '--')
line.plot(a, b2, color = 'black', linestyle = '--')
line.plot(a, b3, color = 'black', linestyle = '-')
b4 = np.maximum(b1, b3)
line.fill_between(a, b2, b4, where= b2>b4, facecolor='red',interpolate=True)
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-1.5, 1.5)
plt.xlim(a.min(), a.max())
plt.text(1.3, 0.1, r'$\alpha$')
plt.text(0.1, 1.3, r'$\beta$')
plt.savefig('./figures/figure_t1q2_4.png')

plt.show()
