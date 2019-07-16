import numpy as np
import copy


class Node:
    # Some Defs
    class OP:
        NONE: int = 0
        ADD: int = 1
        SUB: int = 2
        MUL: int = 3
        MATMUL: int = 4
        ONES_LIKE: int = 5
        ZEROS_LIKE: int = 6
        MATMUL_GRAD1: int = 7
        MATMUL_GRAD2: int = 8
        TO_CONST: int = 9
        RELU: int = 10
        RELU_GRAD: int = 11

    class RUN_FLAG:
        pass

    def __init__(self):
        # Some Values
        self.val: np.ndarray = None
        self.grad: dict = {}
        self.is_constant: bool = False
        self.requires_grad: bool = True
        self.name: str = None
        self.run_flag: RUN_FLAG = None
        self.grad_flag: bool = False
        self.val = None
        self.from_op = Node.OP.NONE
        self.from_num1: Node = None
        self.from_num2: Node = None
        self.from_const = None
        self.this_id = id(self)

    def __repr__(self):
        if not self.is_constant:
            if self.name:
                return "Variable: Name = " + self.name + " , Requires Grad = " + str(self.requires_grad)
            else:
                return "Variable: Name = None , Requires Grad = " + str(self.requires_grad)
        else:
            if self.name:
                return "Constant: Name = " + self.name
            else:
                return "Constant: Name = None"

    def __add__(self, rhs):
        if isinstance(rhs, int) or isinstance(rhs, float):
            ans: Node = Node()
            ans.from_op = Node.OP.ADD
            ans.from_num1 = self
            ans.from_num2 = to_const(rhs, self)
            return ans
        if type(self) != type(rhs):
            raise TypeError("Not the same type!")
        ans: Node = Node()
        ans.from_op: int = Node.OP.ADD
        ans.from_num1: Node = self
        ans.from_num2: Node = rhs
        return ans

    def __radd__(self, rhs):
        if isinstance(rhs, int) or isinstance(rhs, float):
            ans: Node = Node()
            ans.from_op = Node.OP.ADD
            ans.from_num1 = to_const(rhs, self)
            ans.from_num2 = self
            return ans
        if type(self) != type(rhs):
            raise TypeError("Not the same type!")
        ans: Node = Node()
        ans.from_op: int = Node.OP.ADD
        ans.from_num1: Node = rhs
        ans.from_num2: Node = self
        return ans

    def __sub__(self, rhs):
        if isinstance(rhs, int) or isinstance(rhs, float):
            ans: Node = Node()
            ans.from_op = Node.OP.SUB
            ans.from_num1 = self
            ans.from_num2 = to_const(rhs, self)
            return ans
        if type(self) != type(rhs):
            raise TypeError("Not the same type!")
        ans: Node = Node()
        ans.from_op: int = Node.OP.SUB
        ans.from_num1: Node = self
        ans.from_num2: Node = rhs
        return ans

    def __rsub__(self, rhs):
        if isinstance(rhs, int) or isinstance(rhs, float):
            ans: Node = Node()
            ans.from_op = Node.OP.SUB
            ans.from_num1 = to_const(rhs, self)
            ans.from_num2 = self
            return ans
        if type(self) != type(rhs):
            raise TypeError("Not the same type!")
        ans: Node = Node()
        ans.from_op: int = Node.OP.SUB
        ans.from_num1: Node = rhs
        ans.from_num2: Node = self
        return ans

    def __mul__(self, rhs):
        if isinstance(rhs, int) or isinstance(rhs, float):
            ans: Node = Node()
            ans.from_op = Node.OP.MUL
            ans.from_num1 = self
            ans.from_num2 = to_const(rhs, self)
            return ans
        if type(self) != type(rhs):
            raise TypeError("Not the same type!")
        ans: Node = Node()
        ans.from_op: int = Node.OP.MUL
        ans.from_num1: Node = self
        ans.from_num2: Node = rhs
        return ans

    def __rmul__(self, rhs):
        if isinstance(rhs, int) or isinstance(rhs, float):
            ans: Node = Node()
            ans.from_op = Node.OP.MUL
            ans.from_num1 = to_const(rhs, self)
            ans.from_num2 = self
            return ans
        if type(self) != type(rhs):
            raise TypeError("Not the same type!")
        ans: Node = Node()
        ans.from_op: int = Node.OP.MUL
        ans.from_num1: Node = rhs
        ans.from_num2: Node = self
        return ans

    def relu(self):
        ans: Node = Node()
        ans.from_op = Node.OP.RELU
        ans.from_num1 = self
        ans.from_num2 = None
        return ans

    def get_child(self):
        return self.from_num1, self.from_num2, self.from_op

    def __add_backward(self, dest):
        if self.from_num1.requires_grad:
            # Check if have grad before
            if not self.from_num1.grad_flag:
                if self.from_num1.grad.get(dest) is None:
                    self.from_num1.grad[dest] = self.grad[dest] + \
                        zeros_like(self.grad[dest])
            else:
                origin = self.from_num1.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = self.grad[dest]
                origin.from_op = Node.OP.ADD
            self.from_num1.backward(dest)
        if self.from_num2.requires_grad:
            # Check if have grad before
            if not self.from_num2.grad_flag:
                if self.from_num2.grad.get(dest) is None:
                    self.from_num2.grad[dest] = self.grad[dest] + \
                        zeros_like(self.grad[dest])
            else:
                origin = self.from_num2.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = self.grad[dest]
                origin.from_op = Node.OP.ADD
            self.from_num2.backward(dest)

    def __sub_backward(self, dest):
        if self.from_num1.requires_grad:
            # Check if have grad before
            if not self.from_num1.grad_flag:
                if self.from_num1.grad.get(dest) is None:
                    self.from_num1.grad[dest] = self.grad[dest] + \
                        zeros_like(self.grad[dest])
            else:
                origin = self.from_num1.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = self.grad[dest]
                origin.from_op = Node.OP.ADD
            self.from_num1.backward(dest)
        if self.from_num2.requires_grad:
            # Check if have grad before
            if not self.from_num2.grad_flag:
                if self.from_num2.grad.get(dest) is None:
                    self.from_num2.grad[dest] = zeros_like(
                        self.grad[dest])-self.grad[dest]

            else:
                origin = self.from_num2.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = self.grad[dest]
                origin.from_op = Node.OP.SUB
            self.from_num2.backward(dest)

    def __mul_backward(self, dest):
        if self.from_num1.requires_grad:
            # Check if have grad before
            if not self.from_num1.grad_flag:
                if self.from_num1.grad.get(dest) is None:
                    self.from_num1.grad[dest] = self.grad[dest] * \
                        self.from_num2
            else:
                origin = self.from_num1.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = self.grad[dest] * self.from_num2
                origin.from_op = Node.OP.ADD
            self.from_num1.backward(dest)
        if self.from_num2.requires_grad:
            # Check if have grad before
            if not self.from_num2.grad_flag:
                if self.from_num2.grad.get(dest) is None:
                    self.from_num2.grad[dest] = self.grad[dest] * \
                        self.from_num1
            else:
                origin = self.from_num2.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = self.grad[dest] * self.from_num1
                origin.from_op = Node.OP.ADD
            self.from_num2.backward(dest)

    def __matmul_backward(self, dest):
        if self.from_num1.requires_grad:
            # Check if have grad before
            if not self.from_num1.grad_flag:
                if self.from_num1.grad.get(dest) is None:
                    self.from_num1.grad[dest] = matmul_grad1(
                        self.from_num2, self.grad[dest])
            else:
                origin = self.from_num1.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = matmul_grad1(
                    self.from_num2, self.grad[dest])
                origin.from_op = Node.OP.ADD
            self.from_num1.backward(dest)
        if self.from_num2.requires_grad:
             # Check if have grad before
            if not self.from_num2.grad_flag:
                if self.from_num2.grad.get(dest) is None:
                    self.from_num2.grad[dest] = matmul_grad2(
                        self.from_num1, self.grad[dest])
            else:
                origin = self.from_num2.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = matmul_grad2(
                    self.from_num1, self.grad[dest])
                origin.from_op = Node.OP.ADD
            self.from_num2.backward(dest)

    def __relu_backward(self, dest):
        if self.from_num1.requires_grad:
            # Check if have grad before
            if not self.from_num1.grad_flag:
                if self.from_num1.grad.get(dest) is None:
                    self.from_num1.grad[dest] = relu_grad(
                        self.grad[dest], self)
            else:
                origin = self.from_num1.grad[dest]
                tmp = copy.copy(origin)
                origin.from_num1 = tmp
                origin.from_num2 = relu_grad(self.grad[dest], self)
                origin.from_op = Node.OP.ADD
            self.from_num1.backward(dest)

    def backward(self, dest):
        if self.grad_flag:
            return
        self.grad_flag = True
        if self.from_op == Node.OP.NONE:
            return
        elif self.from_op == Node.OP.ADD:
            self.__add_backward(dest)
        elif self.from_op == Node.OP.SUB:
            self.__sub_backward(dest)
        elif self.from_op == Node.OP.MUL:
            self.__mul_backward(dest)
        elif self.from_op == Node.OP.RELU:
            self.__relu_backward(dest)
        elif self.from_op == Node.OP.MATMUL:
            self.__matmul_backward(dest)
        else:
            return

    def clear_flag(self):
        self.grad_flag = False
        if self.from_num1 is not None and self.from_num1.grad_flag:
            self.from_num1.clear_flag()
        if self.from_num2 is not None and self.from_num2.grad_flag:
            self.from_num2.clear_flag()

    def add_name(self, name: str):
        self.name = name


class adexe:

    def __init__(self):
        self.exe_list: list = []
        self.run_code: Node.RUN_FLAG = None

    def __run(self, pos: Node):
        if pos.run_flag and (pos.run_flag is self.run_code):
            return
        if pos.is_constant:
            return
        pos.run_flag = self.run_code

        # Compute
        child1, child2, op = pos.get_child()
        if child1:
            self.__run(child1)
        if child2:
            self.__run(child2)

        if op == Node.OP.NONE:
            pass
        elif op == Node.OP.ADD:
            if child1.val is None or child2.val is None:
                raise ValueError("No enough values for computing!")
            if child1.val.shape != child2.val.shape:
                raise ValueError("Shape not match!")
            pos.val = child1.val + child2.val
        elif op == Node.OP.SUB:
            if child1.val is None or child2.val is None:
                raise ValueError("No enough values for computing!")
            if child1.val.shape != child2.val.shape:
                raise ValueError("Shape not match!")
            pos.val = child1.val - child2.val
        elif op == Node.OP.MUL:
            if child1.val is None or child2.val is None:
                raise ValueError("No enough values for computing!")
            if child1.val.shape != child2.val.shape:
                raise ValueError("Shape not match!")
            pos.val = child1.val * child2.val
        elif op == Node.OP.MATMUL:
            if child1.val is None or child2.val is None:
                raise ValueError("No enough values for computing!")
            if child1.val.ndim != 2 or child2.val.ndim != 2:
                raise ValueError("Must be a 2-dim matrix for matmul!")
            if child1.val.shape[1] != child2.val.shape[0]:
                raise ValueError("Shape not match for matmul!")
            pos.val = np.matmul(child1.val, child2.val)
        elif op == Node.OP.RELU:
            if child1.val is None:
                raise ValueError("No enough values for ReLU!")
            pos.val = 1.0 * (child1.val > 0) * child1.val
        elif op == Node.OP.ONES_LIKE:
            if child1.val is None:
                raise ValueError("No enough values for OnesLike!")
            pos.val = np.ones_like(child1.val)
        elif op == Node.OP.ZEROS_LIKE:
            if child1.val is None:
                raise ValueError("No enough values for ZerosLike!")
            pos.val = np.zeros_like(child1.val)
        elif op == Node.OP.MATMUL_GRAD1:
            if child1.val is None or child2.val is None:
                raise ValueError("No enough values for matmul grad!")
            if child1.val.shape[1] != child2.val.shape[1]:
                raise ValueError("Shape not match for matmul grad!")
            pos.val = np.zeros(
                shape=(child2.val.shape[0], child1.val.shape[0]), dtype=np.float)
            for i in range(child2.val.shape[0]):
                for j in range(child2.val.shape[1]):
                    for k in range(child1.val.shape[0]):
                        pos.val[i, k] += child1.val[k, j] * child2.val[i, j]
        elif op == Node.OP.MATMUL_GRAD2:
            if child1.val is None or child2.val is None:
                raise ValueError("No enough values for matmul grad!")
            if child1.val.shape[0] != child2.val.shape[0]:
                raise ValueError("Shape not match for matmul grad!")
            pos.val = np.zeros(
                shape=(child1.val.shape[1], child2.val.shape[1]), dtype=np.float)
            for i in range(child2.val.shape[0]):
                for j in range(child2.val.shape[1]):
                    for k in range(child1.val.shape[1]):
                        pos.val[k, j] += child1.val[i, k] * child2.val[i, j]
        elif op == Node.OP.TO_CONST:
            pos.val = pos.from_const*np.ones_like(child1.val)
        elif op == Node.OP.RELU_GRAD:
            pos.val = 1.0 * (child2.val > 0) * child1.val

    def run(self, feed_dict: dict = None):
        # feed
        self.run_code = Node.RUN_FLAG()
        for i in feed_dict.items():
            i[0].val = i[1].astype(np.float)
            i[0].run_flag = self.run_code
        # run
        ans: list = []
        for i in self.exe_list:
            self.__run(i)
            ans.append(i.val)
        return tuple(ans)

# Some Functions


def Variable(name: str = None, init_val: np.ndarray = None):
    result = Node()
    result.val = init_val
    result.from_op = Node.OP.NONE
    result.from_num1: Node = None
    result.from_num2: Node = None
    result.name = name
    return result


def Constant(val: np.ndarray = None, name: str = None):
    result = Node()
    result.val = val
    result.is_constant = True
    result.from_op = Node.OP.NONE
    result.from_num1: Node = None
    result.from_num2: Node = None
    result.requires_grad = False
    result.name = name
    return result


def matmul_op(a: Node, b: Node):
    if (not isinstance(a, Node)) or (not isinstance(b, Node)):
        raise TypeError("Not the Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_num2 = b
    ans.from_op = Node.OP.MATMUL
    return ans


def ones_like(a: Node):
    if not isinstance(a, Node):
        raise TypeError("Not a Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_op = Node.OP.ONES_LIKE
    return ans


def zeros_like(a: Node):
    if not isinstance(a, Node):
        raise TypeError("Not a Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_op = Node.OP.ZEROS_LIKE
    return ans


def matmul_grad1(a: Node, b: Node):
    if not isinstance(a, Node) or not isinstance(b, Node):
        raise TypeError("Not the Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_num2 = b
    ans.from_op = Node.OP.MATMUL_GRAD1
    return ans


def matmul_grad2(a: Node, b: Node):
    if not isinstance(a, Node) or not isinstance(b, Node):
        raise TypeError("Not the Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_num2 = b
    ans.from_op = Node.OP.MATMUL_GRAD2
    return ans


def relu(a: Node):
    if not isinstance(a, Node):
        raise TypeError("Not a Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_op = Node.OP.RELU
    return ans

# a: Grad Node. b: Value Node.


def relu_grad(a: Node, b: Node):
    if not isinstance(a, Node) or not isinstance(b, Node):
        raise TypeError("Not the Node Type!")
    ans: Node = Node()
    ans.from_num1 = a
    ans.from_num2 = b
    ans.from_op = Node.OP.RELU_GRAD
    return ans


def to_const(a, b: Node):
    if not isinstance(a, int) and not isinstance(a, float):
        raise TypeError("Not a number!")
    ans: Node = Node()
    ans.from_const = a
    ans.from_num1 = b
    ans.from_op = Node.OP.TO_CONST
    ans.requires_grad = False
    return ans


def Executor(var_list: list):
    exe: adexe = adexe()
    exe.exe_list = var_list.copy()
    return exe


def gradients(dest: Node, sources: list):
    if dest.requires_grad:
        dest.grad[dest] = ones_like(dest)
        dest.backward(dest)
        dest.clear_flag()

    ans: list = []
    for i in sources:
        # Special operation for gradients that not exist.
        if i.grad.get(dest) is None:
            i.grad[dest] = zeros_like(i)
        ans.append(i.grad.get(dest))
    return tuple(ans)
