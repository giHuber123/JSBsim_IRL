import numpy as np
from scipy.integrate import solve_ivp


class AircraftModel:
    def __init__(self):
        self.g = 9.81  # 重力加速度 (m/s^2)

    def dynamics(self, t, state, nx, nz, mu):
        """
        定义微分方程组
        state: [x, y, z, v, gamma, psi]
        控制变量: nx (轴向过载), nz (法向过载), mu (滚转角)
        x,y,z 位置
        v 速度
        gamma 爬升角 >0爬升 <0俯冲
        psi航向角 水平面投影与正北(y轴)夹角
        """

        '''变量, 物理含义, 单位, 缩写, 备注
        "x,y", 水平位置坐标, 米, m, 地面坐标系中的东向和北向
        z, 垂直高度, 米, m, 垂直于地面向上为正
        v, 飞行速度, 米 / 秒, m / s, 
        γ, 轨迹角(Climb angle), 弧度, rad, "向上为正，范围 [−π/2,π/2]"
        ψ, 航向角(Heading angle), 弧度, rad, "范围 [0,2π] 或 [−π,π]"'''

        '''变量, 物理含义, 单位, 缩写, 备注
        nx, 轴向过载, 无量纲, g, 1
        单位等于
        9.81m / s2
        的加速度
        nz, 法向过载, 无量纲, g, 俗称“G力”，平飞时为
        1
        μ, 滚转角(Bank
        angle), 弧度, rad, 决定升力倾斜幅度的角度
        g, 重力加速度, 米 / 秒², m / s², 标准取值 9.80665
        或
        9.81'''
        x, y, z, v, gamma, psi = state

        # 1. 运动学方程 (公式 1)
        dx = v * np.cos(gamma) * np.sin(psi)
        dy = v * np.cos(gamma) * np.cos(psi)
        dz = v * np.sin(gamma)

        # 2. 动力学方程 (公式 2 修正版)
        # v_dot: 速度变化率
        dv = self.g * (nx - np.sin(gamma))

        # gamma_dot: 爬升角变化率 (注意分母应为 v)
        # 避免速度为0导致除以0
        v_safe = max(v, 0.1)
        dgamma = (self.g / v_safe) * (nz * np.cos(mu) - np.cos(gamma))

        # psi_dot: 航向角变化率
        # 当 cos(gamma) 接近 0 时（垂直飞行），航向角定义会失效，这里做简单处理
        cos_gamma = np.cos(gamma)
        if abs(cos_gamma) < 1e-4:
            dpsi = 0
        else:
            dpsi = (self.g * nz * np.sin(mu)) / (v_safe * cos_gamma)

        return [dx, dy, dz, dv, dgamma, dpsi]


# --- 使用示例 ---

# 初始状态: [x=0, y=0, z=5000m, v=250m/s, gamma=0, psi=0]
initial_state = [0, 0, 5000, 250, 0, 0]

# 控制输入: 假设飞机正在进行 5g 的稳定盘旋 (nz=5)，滚转角 60度 (mu = pi/3)
# nx=sin(gamma) 以维持等速飞行
nx_input = 0.0
nz_input = 5.0
mu_input = np.radians(60)

model = AircraftModel()
t_span = (0, 100)  # 模拟10秒
t_eval = np.linspace(0, 100, 100)  # 输出时间点

sol = solve_ivp(
    model.dynamics,
    t_span,
    initial_state,
    args=(nx_input, nz_input, mu_input),
    t_eval=t_eval,
    method='RK45'
)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 接续之前的计算代码 ---

# 提取积分结果
x_path = sol.y[0, :]
y_path = sol.y[1, :]
z_path = sol.y[2, :]

# 创建画布
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 实线轨迹
ax.plot(x_path, y_path, z_path, label='Flight Path', color='b', lw=2)

# 绘制在底面 (X-Y平面) 的投影线，方便观察转弯半径
ax.plot(x_path, y_path, np.min(z_path) - 10, color='gray', linestyle='--', alpha=0.5, label='Ground Projection')

# 标记起点和终点
ax.scatter(x_path[0], y_path[0], z_path[0], color='green', s=100, label='Start')
ax.scatter(x_path[-1], y_path[-1], z_path[-1], color='red', s=100, label='End')

# 设置轴标签
ax.set_xlabel('East (x) [m]')
ax.set_ylabel('North (y) [m]')
ax.set_zlabel('Altitude (z) [m]')
ax.set_title('3D Aircraft Trajectory Simulation')

# 保持坐标轴比例一致（防止转弯变成椭圆）
# 注意：在旧版 matplotlib 中需要手动计算边界，新版可用 set_box_aspect
max_range = np.array([x_path.max()-x_path.min(), y_path.max()-y_path.min(), z_path.max()-z_path.min()]).max() / 2.0
mid_x = (x_path.max()+x_path.min()) * 0.5
mid_y = (y_path.max()+y_path.min()) * 0.5
mid_z = (z_path.max()+z_path.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.legend()
plt.show()