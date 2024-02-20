import sympy as sp

x = sp.symbols('x')
a = sp.Symbol('a')
b = sp.Symbol('b')
c = sp.Symbol('c')
C = sp.Symbol('C')  # 定义常数 C

u = a + b*x + c*x**2
du = sp.diff(u, x)
integral = (1 / (2*c)) * sp.asinh(u) + C

# 使用 sympy.integrate() 函数计算不定积分
result = sp.integrate(integral, x)

# 打印结果
print(result(0))