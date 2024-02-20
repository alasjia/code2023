import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 1.4, 100)
# y = np.sqrt(1 + np.tan(x)**2) * (0.936 * np.tan(x) + 0.351)

# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('sqrt(1+tan(x)^2)*(0.5*tan(x)+0.5)')
# plt.title('Graph of sqrt(1+tan(x)^2)*(0.5*tan(x)+0.5)')
# plt.grid(True)
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

temp1 = np.sqrt(1.8**2+4.8**2)/2
sin0 = 0.9/temp1 
cos0 = 2.4/temp1

x = np.linspace(0, np.tan(np.pi/3), 100)
y = np.sqrt(1 + x**2) * (cos0 * x + sin0)


plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sqrt(1+x^2)*(0.5*x+0.5)')
plt.title('Graph of sqrt(1+x^2)*(0.5*x+0.5)')
plt.grid(True)

# 计算直线的函数表达式
x_points = [0, np.tan(np.pi/3)]
y_points = [np.sqrt(1 + x_points[0]**2) * (0.936 * x_points[0] + 0.351), np.sqrt(1 + x_points[1]**2) * (0.936 * x_points[1] + 0.351)]
coefficients = np.polyfit(x_points, y_points, 1)
poly = np.poly1d(coefficients)
plt.plot(x, poly(x), label=f'{coefficients[0]:.2f}x + {coefficients[1]:.2f}')
plt.legend()


# 计算最大差值并进行标记
max_diff = np.max(np.abs(y - poly(x)))
max_index = np.argmax(np.abs(y - poly(x)))
plt.axvline(x[max_index], color='r', linestyle='--')  # 沿y轴画一条竖线
plt.annotate(f'Max diff: {max_diff:.2f}', xy=(x[max_index], y[max_index] + 0.3), xytext=(x[max_index]+0.2, y[max_index]+0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()