import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#线段容器
with open('/home/pc/桌面/tridexel-master（复件）/tridexel-master/build/SEintersection_pointsXY.txt', 'r') as fileXY:
    linesXY = fileXY.readlines()
with open('/home/pc/桌面/tridexel-master（复件）/tridexel-master/build/SEintersection_pointsYZ.txt', 'r') as fileYZ:
    linesYZ = fileYZ.readlines()
with open('/home/pc/桌面/tridexel-master（复件）/tridexel-master/build/SEintersection_pointsXZ.txt', 'r') as fileXZ:
    linesXZ = fileXZ.readlines()


with open('/home/pc/桌面/tridexel-master（复件）/tridexel-master/build/TEintersection_pointsXZ.txt', 'r') as fileXYTE:
    linesXYTE = fileXYTE.readlines()


# 初始化存储线段端点的列表
segments = []

# 解析每两行，将线段端点存储为元组
# for i in range(0, len(linesXY), 2):
#     x1, y1, z1 = map(float, linesXY[i].split())
#     x2, y2, z2 = map(float, linesXY[i + 1].split())
#     segments.append(((x1, y1, z1), (x2, y2, z2)))

# for i in range(0, len(linesYZ), 2):
#     x1, y1, z1 = map(float, linesYZ[i].split())
#     x2, y2, z2 = map(float, linesYZ[i + 1].split())
#     segments.append(((x1, y1, z1), (x2, y2, z2)))

# for i in range(0, len(linesXZ), 2):
#     x1, y1, z1 = map(float, linesXZ[i].split())
#     x2, y2, z2 = map(float, linesXZ[i + 1].split())
#     segments.append(((x1, y1, z1), (x2, y2, z2)))

for i in range(0, len(linesXYTE), 2):
    x1, y1, z1 = map(float, linesXYTE[i].split())
    x2, y2, z2 = map(float, linesXYTE[i + 1].split())
    segments.append(((x1, y1, z1), (x2, y2, z2)))


# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制每个线段
for segment in segments:
    x_values = [segment[0][0], segment[1][0]]
    y_values = [segment[0][1], segment[1][1]]
    z_values = [segment[0][2], segment[1][2]]
    ax.plot(x_values, y_values, z_values, color='blue')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# 显示图形
plt.show()