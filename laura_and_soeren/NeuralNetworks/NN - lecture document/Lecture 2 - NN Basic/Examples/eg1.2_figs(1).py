#
# Chapter 1, Example 2
#

import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

x = np.arange(-3.0, 3.1, 0.1)

y1 = -x -1
y2 = -0.5*x + 0.5
y3 = x - 2


# line1 = plt.figure(1)
line1 = plt.subplot(1, 1, 1)
line1.plot(x, y1, color = 'black', linestyle = '--')
line1.fill_between(x, y1, 2.5, facecolor='yellow')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.ylim(-3.0, 3.0)
plt.xlim(x.min(), x.max())
plt.text(-1.5, -1.5, '$y_1 = 0$')
plt.text(1.0, 1.0, '$y_1 = 1$')
plt.text(2.7, 0.2, '$x_1$')
plt.text(0.2, 2.7, '$x_2$')
plt.savefig('./figures/1.2_1.png')


plt.figure(2)
line2 = plt.subplot(1, 1, 1)
line2.plot(x, y2,color = 'black', linestyle = '-')
line2.fill_between(x, y2, -3.0, facecolor='green')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim(x.min(), x.max())
plt.ylim(-3.0, 3.0)
plt.text(-1.0, -1.0, '$y_2 = 0$')
plt.text(1.0, 1.0, '$y_2 = 1$')
plt.text(2.7, 0.2, '$x_1$')
plt.text(0.2, 2.7, '$x_2$')
plt.savefig('./figures/1.2_2.png')

plt.figure(3)
line3 = plt.subplot(1, 1, 1)
line3.fill_between(x, y3, 3.0, facecolor='blue')
line3.plot(x, y3, color = 'black', linestyle = '--')
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.xlim(x.min(), x.max())
plt.ylim(-3.0, 3.0)
plt.text(1.0, -2.0, '$y_3 = 0$')
plt.text(-1.0, 1.0, '$y_3 = 1$')
plt.text(2.7, 0.2, '$x_1$')
plt.text(0.2, 2.7, '$x_2$')
plt.savefig('./figures/1.2_3.png')

plt.figure(4)
line = plt.subplot(1, 1, 1)
line.plot(x, y1, color = 'black', linestyle = '--')
line.plot(x, y2, color = 'black', linestyle = '-')
line.plot(x, y3, color = 'black', linestyle = '--')
y4 = np.maximum(y1, y3)
line.fill_between(x, y2, y4, where= y2>y4, facecolor='red',interpolate=True)
plt.xlim(-3.0, 3.0)
plt.ylim(-3.0, 3.0)
ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.text(-1., 0.25, '$y = 1$')
plt.text(1.0, 1.0, '$y = 0$')
plt.savefig('./figures/1.2_4.png')

plt.show()
