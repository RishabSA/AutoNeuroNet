import autoneuronet as ann
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def objective_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * x**2 + 5 * y**2


def objective_ann(x: ann.Var, y: ann.Var) -> ann.Var:
    return 2 * x**2 + 5 * y**2


def run_gradient_descent(x0: float, y0: float, lr: float, steps: int):
    xs, ys, zs = [], [], []

    x = ann.Var(x0)
    y = ann.Var(y0)

    for step in range(steps):
        z = objective_ann(x, y)
        z.setGrad(1.0)
        z.backward()

        xs.append(float(x.val))
        ys.append(float(y.val))
        zs.append(float(z.val))

        # Gradient Descent
        x = ann.Var(x.val - lr * x.grad)
        y = ann.Var(y.val - lr * y.grad)

    return np.stack((np.array(xs), np.array(ys), np.array(zs)), 1)


if __name__ == "__main__":
    xg = np.linspace(-4.0, 4.0, 80)
    yg = np.linspace(-4.0, 4.0, 80)

    X, Y = np.meshgrid(xg, yg)
    Z = objective_np(X, Y)

    path = run_gradient_descent(x0=3.2, y0=-3.0, lr=1e-2, steps=100)
    xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("AutoNeuroNet: 3D Gradient Descent")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")

    ax.view_init(elev=30, azim=150)
    ax.set_box_aspect((1, 1, 0.6))

    ax.plot_surface(
        X, Y, Z, cmap="coolwarm", alpha=0.75, rstride=2, cstride=2, linewidth=0
    )

    (line,) = ax.plot([], [], [], color="#ff5533", linewidth=2)
    (point,) = ax.plot([], [], [], "o", color="#ff5533", markersize=6)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])

        point.set_data([], [])
        point.set_3d_properties([])

        return line, point

    def update(i):
        line.set_data(xs[: i + 1], ys[: i + 1])
        line.set_3d_properties(zs[: i + 1])

        point.set_data([xs[i]], [ys[i]])
        point.set_3d_properties([zs[i]])

        return line, point

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(xs),
        interval=120,
        blit=True,
    )

    anim.save("demos/gradient_descent.gif", writer=animation.PillowWriter(fps=24))

    plt.tight_layout()
    plt.show()
