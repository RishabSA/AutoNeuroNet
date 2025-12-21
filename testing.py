import torch

if __name__ == "__main__":
    X = torch.tensor([[1, 1, 1], [2, 2, 2]])
    print(f"X: {X}")

    Y = X**2
    print(f"Y: {Y}")
