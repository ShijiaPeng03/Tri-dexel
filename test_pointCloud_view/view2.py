
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取文件内容并解析点云数据
points = []
with open("/home/psj/Desktop/tridexel-master/tridexel-master/build/KnifeInsertPoint", "r") as file:
    for line in file:
        x, y, z = map(float, line.split())
        points.append((x, y, z))

# 创建Matplotlib图形对象并绘制点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取点的坐标分量
x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]

# 绘制点云
ax.scatter(x, y, z, c='b', marker='o')  # 使用蓝色圆点表示点

# 设置图形标题和坐标轴标签
ax.set_title('Point Cloud')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()




