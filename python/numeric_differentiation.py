def quadratic(x):
    return x**2


def forward_numeric_differentiation(f, x):
    h = 1e-8
    return (f(x + h) - f(x)) / h


def central_numeric_differentiation(f, x):
    h = 1e-5
    return (f(x + h) - f(x - h)) / (2.0 * h)


if __name__ == "__main__":
    x = 5

    print(f"f({x}) = {quadratic(x)}")
    print(f"forward f'({x}) = {forward_numeric_differentiation(quadratic, x)}")
    print(f"central f'({x}) = {central_numeric_differentiation(quadratic, x)}")
