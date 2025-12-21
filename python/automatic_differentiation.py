class Var:
    def __init__(self, value: float):
        self.value = value
        self.children = []
        self.grad_value = None

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        z = Var(self.value + other.value)

        # dz/d_self = 1
        self.children.append((1.0, z))

        # dz/d_other = 1
        other.children.append((1.0, z))

        return z

    def __sub__(self, other):
        z = Var(self.value - other.value)

        # dz/d_self = 1.0
        self.children.append((1.0, z))

        # dz/d_other = -1.0
        other.children.append((-1.0, z))

        return z

    def __mul__(self, other):
        z = Var(self.value * other.value)

        # dz/d_self = other.value
        self.children.append((other.value, z))

        # dz/d_other = self.value
        other.children.append((self.value, z))

        return z

    def __pow__(self, power):
        z = Var(self.value**power)

        # dz/d_self = power * self.value ** (power - 1)
        self.children.append((power * self.value ** (power - 1), z))

        return z

    def __neg__(self):
        z = Var(-self.value)

        # dz/d_self = -1.0
        self.children.append((-1.0, z))

        return z

    def __truediv__(self, other):
        if other.value == 0:
            raise ZeroDivisionError

        z = Var(self.value / other.value)

        # dz/d_self = 1.0 / other.value
        self.chldren.append((1.0 / other.value, z))

        # dz/d_other = -self.value / other.value^2
        other.children.append((-self.value / (other.value**2), z))

        return z

    def grad(self):
        if self.grad_value is None:
            # Compute derivative using the chain rule
            self.grad_value = sum(weight * var.grad() for weight, var in self.children)

        return self.grad_value


if __name__ == "__main__":
    # Reverse-Mode Automatic Differentiation

    x = Var(5)
    y = Var(10)

    f = y * x**2
    f.grad_value = 1.0
    print(f)

    print(f"df/dx = {x.grad()}")  # 100
    print(f"df/dy = {y.grad()}")  # 25
