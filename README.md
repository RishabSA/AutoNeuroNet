# Automatic Differentiation

## Setup

My implementation uses C++ for all derivatives, matrices, and neural networks. PyBind11 is used to make these accessible in Python.

To build the necessary C++ classes for use in Python run:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

The C++ dependencies can then be accessed from the build/ folder and imported in any Python files.

Resources I used as reference:

- [What's Automatic Differentiation? - HuggingFace](https://huggingface.co/blog/andmholm/what-is-automatic-differentiation)
- [Differentiate Automatically](https://comp6248.ecs.soton.ac.uk/handouts/autograd-handouts.pdf)

---

- Computing **derivatives** and **gradients** is very important for **machine learning**
  - Needed for **gradient-based optimization algorithms**
- Methods of **computing derivatives** **through code:**
  - **Numeric Differentiation**
  - **Symbolic Differentiation**
  - **Automatic Differentiation**

## Numeric Differentiation

- **Numeric Differentiation** uses the **limit definition** of a **derivative** to compute the **derivative** at a **point**
  - To evaluate the **derivative** of a function $f$ at the point $x$, compute the **instantaneous rate of change** at the point $x$ as $h$ approaches $0$
    - **Version** of the **slope formula** $\frac{\text{rise}}{\text{run}}$
  - **Forward numeric differentiation**
    $$
    \frac{df}{dx} = \lim_{h \to 0}\frac{f(x+h) - f(x)}{h}
    $$
    - Classic **limit-definition** of the **derivative** with **finite differences**
  - **Central numeric differentiation**
    $$
    \frac{df}{dx} = \lim_{h \to 0}\frac{f(x+h) - f(x - h)}{2h}
    $$
    - **Combination** of the **forward** and **backward difference** for **differentiation**
      - **More stable** than **forward numeric differentiation** as it reduces **any** **truncation error** by a factor of $2$

### Issues

- **Requires multiple function evaluations** to compute the **derivative**
- Results can change with the choice of $h$
  - $h$ is set to a **very small floating point value**, and is limited by **floating point memory constraints**
- Results can suffer from **round-off** and **truncation errors**
  - **Round-off error** is the **error** due to **inaccuracies** with **floating point values in computers**
    - Increasing the **precision** of $h$ **increases hardware constraints** due to **more memory requirements**

## Symbolic Differentiation

- **Symbolic Differentiation** is another approach to **computing derivatives**
  - Converts **symbolic mathematical expressions** to their **symbolic derivative expressions**
  - Uses the **derivative rules** of **Calculus**, such as the **product rule**, **chain rule**, **quotient rule**, **trig derivatives**, etc
  - `sympy` is a **python library** designed for **symbolic math** in **python**, which has **symbolic differentiation**

### Issues

- With very complex functions, the **derivative expressions grow very fast,** become **extremely long**, and **require extra computations**
  - Taking the **derivative** of a **product** with the **product rule** grows the number of **terms** and **computations**
    $$
    \frac{d}{dx}f(x)g(x)=f^\prime(x)g(x)+g^\prime(x)f(x)
    $$
  - When the **terms grow** in the **original expression**, the **expression** of the **derivative** will **grow even faster**
- **Limited** to **closed-form, mathematical expressions**
  - **Symbolic Differentiation** cannot represent **derivatives** when working with **control flow changes** (if statements, loops, etc)

## Automatic Differentiation

- **Automatic Differentiation** **(AD)** uses the **chain rule** on the **sequence** of **primitive operations** that are being **computed**
  - **Expresses** **composite functions** in the **variables** and **elementary operations** that form them
  - **Runs** the **chain rule** on each **addition**, **multiplication**, **division**, **sine**, **exponential**, **natural log**, etc
    - Derivatives are easy to compute
  - **Computes derivatives alongside** each **function evaluation**
- **AD represents** a **program** as a **graph of primitive operations**, then **propagates derivatives through it** with the c**hain rule**
  - **Stores information** for **forward computations** that can be reused for **backward propagation**

### Chain Rule and Computational Graph

- **Automatic Differentiation** applies the **chain rule** **locally** at **each node** and **propagates**
  - For a **function composition** $y = f(g(x)$, the **derivative** using the **chain rule** is given as:
    $$
    \frac{dy}{dx}=\frac{df}{dg} \cdot \frac{dg}{dx} = f^\prime(g(x)) \cdot g^\prime(x)
    $$
  - When working with **multiple variables**, take the **partial derivative** of the **function** with respect to **each of the variables**
    - For a **multi-variable function composition $z = f(g(x, y)$**, the **partial derivatives with respect to $x$ and $y$** is given as:
      $$
      \frac{\partial z}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}
      $$
      $$
      \frac{\partial z}{\partial y} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial y}
      $$
- In general, the **chain rule** can be expressed as:
  $$
  \frac{\partial w}{\partial t} = \sum_{i=1}^N\frac{\partial w}{\partial u_i}\frac{\partial u_i}{\partial t}
  $$
  - Where:
    - $w$ is an **output variable**
    - $t$ is an **input variable**
    - $u_i$ is an **intermediate variable**
- These **functions** can be **represented** as a **computation graph,** **Directed Acyclic Graph (DAG)**, **evaluation trace**, or a **composition** of **primitive operations**
  - **Nodes** represent **input** **variables**, **intermediate variables**, and **output variables**
    - These variables are called **primals** and are denoted $v_i$
  - **Edges** represent **operations** or the **computation hierarchy** between **inputs** and **outputs**
- For an **example function** $f(x,y) = xy + \sin(x)$, the **computations** are:
  1. $v_2=x_0 x_1$
  2. $v_3=\sin(x)$
  3. $y=v_2+v_3$
- The **forward computation primal evaluation trace** is given below for $x_0 = 2$ and $x_1 = 4$
  | **Forward Primal Trace** | **Output** |
  | ------------------------ | ----------------- |
  | $v_0=x_0$ | $2$ |
  | $v_1=x_1$ | $4$ |
  | $v_2=v_0v_1$ | $2(4)=8$ |
  | $v_3=\sin(x)$ | $0.0349$ |
  | $v_4=v_2+v_3$ | $8+0.0349=8.0349$ |
  | $y=v_4$ | $8.0349$ |
- To make the **differentiation** process **automatic**, we define the **basic differentiation rules** for **all operations**
- **Automatic Differentiation** comes in $2$ **modes**: **Forward-Mode AD** and **Reverse-Mode AD**

### Forward-Mode Automatic Differentiation

- **Forward-Mode Automatic differentiation** **propagates (value, derivative) pairs forward**
  - **Stores** **(value, derivative) pairs** at each **computation**
- For example:
  $$
  v_2 = x_0 x_1 \\ \frac{\partial v_2}{\partial t} = x_0\frac{\partial x_1}{\partial t} + x_1\frac{\partial x_0}{\partial t}
  $$
  $$
  v_3=\sin(v_0)\\ \frac{\partial v_3}{\partial t} = \cos(v_0)\frac{\partial v_0}{\partial t}
  $$
  $$
  y=v_2+v_3\\ \frac{\partial y}{\partial t} = \frac{\partial v_2}{\partial t} + \frac{\partial v_3}{\partial t}
  $$
  - Where:
    - To compute the **partial derivative with respect to $x_0$,** we can substitute $t=v_0$ into the above
    - o compute the **partial derivative with respect to $x_1$,** we can substitute $t=v_1$ into the above
- The **forward mode primal and derivative trace** is given below for $x_0=2$ and $x_1=4$ to compute $\frac{\partial y}{\partial x_0}$
  | **Forward Primal Trace** | **Output** | **Forward Derivative Trace** | **Output** |
  | ------------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------- |
  | $v_0=x_0$ | $2$ | $\frac{\partial v_0}{\partial v_0} = \frac{\partial v_0}{\partial x_0}$ | $1$ |
  | $v_1=x_1$ | $4$ | $\frac{\partial v_1}{\partial v_0} = \frac{\partial v_1}{\partial x_0}$ | $0$ |
  | $v_2=v_0v_1$ | $2(4)=8$ | $\frac{\partial v_2}{\partial v_0} = v_0\frac{\partial v_1}{\partial v_0} + v_1\frac{\partial v_0}{\partial v_0}$ | $2 \cdot 0 + 4 \cdot 1 = 4$ |
  | $v_3=\sin(v_0)$ | $0.0349$ | $\frac{\partial v_3}{\partial v_0} = \cos(v_0)\frac{\partial v_0}{\partial v_0}$ | $0.99 \cdot 1 = 0.99$ |
  | $v_4=v_2+v_3$ | $8+0.0349=8.0349$ | $\frac{\partial v_4}{\partial v_0} = \frac{\partial v_2}{\partial v_0} + \frac{\partial v_3}{\partial v_0}$ | $4 + 0.99 = 4.99$ |
  | $y=v_4$ | $8.0349$ | $\frac{\partial y}{\partial v_0} = \frac{\partial v_4}{\partial v_0}$ | $4.99$ |
- The **forward mode primal and derivative trace** is given below for $x_0=2$ and $x_1=4$ to compute $\frac{\partial y}{\partial x_1}$
  | **Forward Primal Trace** | **Output** | **Forward Derivative Trace** | **Output** |
  | ------------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------- |
  | $v_0=x_0$ | $2$ | $\frac{\partial v_0}{\partial v_1} = \frac{\partial v_0}{\partial x_1}$ | $0$ |
  | $v_1=x_1$ | $4$ | $\frac{\partial v_1}{\partial v_1} = \frac{\partial v_1}{\partial x_1}$ | $1$ |
  | $v_2=v_0v_1$ | $2(4)=8$ | $\frac{\partial v_2}{\partial v_1} = v_0\frac{\partial v_1}{\partial v_1} + v_1\frac{\partial v_0}{\partial v_1}$ | $2 \cdot 1 + 4 \cdot 0 = 2$ |
  | $v_3=\sin(v_0)$ | $0.0349$ | $\frac{\partial v_3}{\partial v_1} = \cos(v_0)\frac{\partial v_0}{\partial v_1}$ | $0.99 \cdot 0 = 0$ |
  | $v_4=v_2+v_3$ | $8+0.0349=8.0349$ | $\frac{\partial v_4}{\partial v_1} = \frac{\partial v_2}{\partial v_1} + \frac{\partial v_3}{\partial v_1}$ | $2+0=2$ |
  | $y=v_4$ | $8.0349$ | $\frac{\partial y}{\partial v_1} = \frac{\partial v_4}{\partial v_1}$ | $2$ |
- **Forward-Mode AD** is **easy to implement**
- It has a **big issue**: **for every variable** that the **derivative is computed for**, the entire **program** has to be **run again**
  - With **networks** with **many parameters**, this is **incredibly inefficient**
  - In the above example, we had to **repeat** the **process** for both $x_0$ and $x_1$
  - **Forward-Mode AD** works best when $n$ **parameters** is **small** and $m$ **outputs** is **large** (lots of outputs depending on few parameters)
- The solution is **Reverse-Mode Automatic Differentiation**

### **Reverse-Mode Automatic Differentiation**

- **Reverse-Mode Automatic Differentiation** is **more efficient** than **Forward-Mode AD**
  - Works well when working with **many parameters** like in **neural networks**
- Instead of storing derivatives of intermediate primal variables with respect to inputs, **Reverse-Mode AD stores derivatives of the final output with respect to each intermediate primal**
- **D**efines **adjoints** $\bar{v}_i$ as the **partial derivative** of an **output** $y$ **with respect to** an **intermediate** **primal variable $v_i$**
  $$
  \bar{v}_i= \frac{\partial y}{\partial v_i}
  $$
- **Reverse-Mode AD** performs a **forward pass** by running the **function forward**
  - During this process, it **computes** **intermediate variables**
  - Instead of computing **adjoints alongside** **primals** like in **Forward-Mode**, **Reverse-Mode** stores any **necessary dependencies** required for the **derivative computation** of $\bar{v}_i$ in the **computational graph**
    - Uses the **derivatives** of **elementary operations**, the **chain rule**, and **cached dependencies** from the forward pass to compute **adjoints**
    - **Adjoints** are computed through the **reverse pass** or **backpropagation**, **starting** from an **output variable** and **ending** with all **input variables**
      - **Derivatives** are computed in a **reversed order**
  - Initialize $\bar{y} = \frac{\partial y}{\partial y} = 1$ and **backpropagate derivatives** with the **chain rule**
    - At the end of this process, we get $\bar{x}_i = \frac{\partial y}{\partial x_i}$ for **all inputs** in a **single backward pass**
- The **forward mode primal and reverse mode adjoint trace** is given below for $x_0=2$ and $x_1=4$
  | **Forward Primal Trace** | **Output** | **Reverse Adjoint Trace** | **Output** |
  | ------------------------ | ----------------- | --------------------------------------------------------------------- | ---------------------------------- |
  | $v_0=x_0$ | $2$ | $\bar{v}_0=\bar{x}_0=\bar{v}_3 \cdot \cos(v_0) + \bar{v}_2 \cdot v_1$ | $1 \cdot 0.999 +1 \cdot 4 = 4.999$ |
  | $v_1=x_1$ | $4$ | $\bar{v}_1=\bar{x}_1=\bar{v}_4 \cdot 1 + \bar{v}_2 \cdot v_0$ | $1 \cdot 1 + 1\cdot 2=3$ |
  | $v_2=v_0v_1$ | $2(4)=8$ | $\bar{v}_2=\bar{v}_4 \cdot 1$ | $1 \cdot 1= 1$ |
  | $v_3=\sin(v_0)$ | $0.0349$ | $\bar{v}_3 = \bar{v}_4 \cdot 1$ | $1 \cdot 1 = 1$ |
  | $v_4=v_2+v_3$ | $8+0.0349=8.0349$ | $\bar{v}_4=\bar{y}$ | $1$ |
  | $y=v_4$ | $8.0349$ | $\bar{y}$ | $1$ |
- **Basic derivative rules** are applied in the **reverse pass**
- **Reverse-Mode AD** works best when $n$ **parameters** is **large** and $m$ **outputs** is **small** (few outputs depending lots of parameters)
  - Used in the **training of neural networks**
  - **Reverse-Mode Automatic Differentiation** is the best option for **gradient-based optimization**
    - Only **one reverse pass** is **needed** for **each step** of **gradient descent**
