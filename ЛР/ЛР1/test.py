import torch

# 创建一个需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)

# 使用 x 计算 y = x^2
y = x ** 2

# 创建一个新的张量，该张量与 y 共享数据但不需要梯度
z = y.detach().requires_grad_(True)

# 使用新的张量 z 计算 y^2
output = z ** 2

# 计算梯度
output.backward()

# 输出梯度
print("Gradient of x:", x.grad)  # 输出应为 8.0，因为 y = x^2，dy/dx = 2*x = 2*2 = 4；再次求导，d(y^2)/dx = 2*y*dy/dx = 2*(x^2)*4 = 8
