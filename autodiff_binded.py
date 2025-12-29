from build.Var import Var

x0 = Var(5)
x1 = Var(10)

z = x0**2
y = z * x1

y.gradVal = 1
y.backward()

print(f"y = {y}")
print(f"x0 = {x0}")
print(f"x1 = {x1}")
