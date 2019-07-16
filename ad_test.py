import autodiff as ad
import numpy as np
import matplotlib.pyplot as plt

# Hyper Parameters
LR = 0.0001
EPOCH = 500

x_val = np.linspace(-5, 5, 50)
y_val = x_val * x_val + np.random.rand(50)*0.5

x = ad.Variable(name='x')
y_ = ad.Variable(name='y_')
w1 = ad.Variable(name='w1', init_val=np.random.rand(1, 10))
b1 = ad.Variable(name='b1', init_val=np.random.rand(1, 10))
w2 = ad.Variable(name='w2', init_val=np.random.rand(10, 10))
b2 = ad.Variable(name='b2', init_val=np.random.rand(1, 10))
w3 = ad.Variable(name='w3', init_val=np.random.rand(10, 1))
b3 = ad.Variable(name='b3', init_val=np.random.rand(1, 1))

fc1 = ad.relu(ad.matmul_op(x, w1) + b1)
fc2 = ad.relu(ad.matmul_op(fc1, w2) + b2)
y = ad.matmul_op(fc2, w3) + b3

loss = (y_ - y) * (y_ - y)

w1_grad, b1_grad, w2_grad, b2_grad, w3_grad, b3_grad = ad.gradients(
    loss, [w1, b1, w2, b2, w3, b3])

executor = ad.Executor(
    [loss, w1_grad, b1_grad, w2_grad, b2_grad, w3_grad, b3_grad])

for epoch in range(EPOCH):
    for i in range(x_val.shape[0]):
        loss_val, w1_grad_val, b1_grad_val, w2_grad_val, b2_grad_val, w3_grad_val, b3_grad_val = executor.run(
            feed_dict={x: x_val[i].reshape([1, 1]), y_: y_val[i].reshape([1, 1])})

        print("EPOCH:", epoch, "STEP:", i, "LOSS:", loss_val)

        # Optimize
        w1.val = w1.val - LR * w1_grad_val
        b1.val = b1.val - LR * b1_grad_val
        w2.val = w2.val - LR * w2_grad_val
        b2.val = b2.val - LR * b2_grad_val
        w3.val = w3.val - LR * w3_grad_val
        b3.val = b3.val - LR * b3_grad_val


executor = ad.Executor([y])
ans = []
for i in range(x_val.shape[0]):
    y_val_ans, = executor.run(
        feed_dict={x: x_val[i].reshape([1, 1]), y_: y_val[i].reshape([1, 1])})
    ans.append(y_val_ans.reshape(1))

ans = np.array(ans)

plt.scatter(x_val, y_val, color='g')
plt.plot(x_val, ans, color='b')
plt.show()
