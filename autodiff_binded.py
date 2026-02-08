from build.autoneuronet import Var, Matrix


def autoneuronet_var_demo():
    x0 = Var(5)
    x1 = Var(10)

    z = x0**2
    y = z * x1

    y.grad = 1
    y.backward()

    print(f"y = {y}")
    print(f"x0 = {x0}")
    print(f"x1 = {x1}")


def autoneuronet_matrix_demo():
    mat1 = Matrix(5, 2)
    mat2 = Matrix(2, 4)

    mat1.randomInit()

    for x in range(2):
        for y in range(4):
            mat2[x, y] = 2 * x + y

    mat3 = mat1 @ mat2

    print(f"mat1 = {mat1}")
    print(f"mat2 = {mat2}")
    print(f"mat3 = {mat3}")


if __name__ == "__main__":
    autoneuronet_var_demo()
    # autoneuronet_matrix_demo()
