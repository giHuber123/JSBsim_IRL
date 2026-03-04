import numpy as np
import matplotlib.pyplot as plt

class AircraftModel:
    def __init__(self):
        self.g = 9.81  # 重力加速度 (m/s^2)
        self.nx_max = 4
        self.nx_min = 0
        self.ny_max = 7
        self.ny_min = -5
        self.gamma_max = np.pi
        self.gamma_min = np.deg2rad(0)

    def dynamics(self, t, state, nx, ny, gamma):

        # nx = np.clip(nx, 0, 4)
        # ny = np.clip(ny, -5, 7)
        # gamma = np.clip(gamma, 0, np.pi)
        '''
        x z y:纬度 经度 高度
        v theta psi:速度 航迹倾斜角 航向角
        nx ny gamma:切向过载 法向过载 绕速度轴滚转角 [0,4] [-5,7] [0,pi]
        '''
        x, y, z, v, theta, psi = state

        # 1. 运动学方程 (公式 1)
        dx = v * np.cos(theta) * np.cos(psi)
        dy = v * np.sin(theta)
        dz = -v * np.cos(theta) * np.sin(psi)

        # 2. 动力学方程 (公式 2 修正版)
        # v_dot: 速度变化率
        dv = self.g * (nx - np.sin(theta))
        dtheta = self.g / max(v, 0.1) * (ny * np.cos(gamma) - np.cos(theta))

        # psi_dot: 航向角变化率
        if abs(np.cos(theta)) < 1e-4:
            dpsi = 0
        else:
            dpsi = -self.g / (max(v, 0.1) * np.cos(theta)) * ny * np.sin(gamma)
        # gamma_dot: 爬升角变化率 (注意分母应为 v)

        return [dx, dy, dz, dv, dtheta, dpsi]

    def initiate(self):
        x = 0 #m
        z = 0 #m
        y = 5000 #m
        v = 400 #m/s
        theta = np.deg2rad(0) #°
        psi = np.deg2rad(0) #°

        return np.array([x, y, z, v, theta, psi])

    def get_action(self, action_id):
        # 预计算常数
        bank = np.arccos(1 / 8)
        nx_max = 1.2
        nx_min = -0.8
        ny_max = 9.0
        ny_min = -3.0

        # 定义动作字典: {id: (nx, ny, gamma)}
        # actions = {
        #     0: (0, 1, 0),  # 匀速前飞
        #     1: (1, 1, 0),  # 加速前飞
        #     2: (-1, 1, 0),  # 减速前飞
        #
        #     3: (0, 8, -bank),  # 匀速右转
        #     4: (1, 8, -bank),  # 加速右转
        #     5: (-1, 8, -bank),  # 减速右转
        #
        #     6: (0, 8, bank),  # 匀速左转
        #     7: (1, 8, bank),  # 加速左转
        #     8: (-1, 8, bank),  # 减速左转
        #
        #     9: (0.5, 8, 0),  # 匀速爬升
        #     10: (1, 8, 0),  # 加速爬升
        #     11: (-1, 8, 0),  # 减速爬升
        #
        #     12: (0, 0.5, 0),  # 匀速俯冲
        #     13: (1, 0.5, 0),  # 加速俯冲
        #     14: (-1, 0.5, 0),  # 减速俯冲
        # }

        actions = {
            0: (0, 1, 0),  # 匀速前飞
            1: (nx_max, 1, 0),  # 加速前飞
            2: (nx_min, 1, 0),  # 减速前飞

            3: (0, 8, -bank),  # 匀速右转
            4: (nx_max, 8, -bank),  # 加速右转
            5: (nx_min, 8, -bank),  # 减速右转

            6: (0, 8, bank),  # 匀速左转
            7: (nx_max, 8, bank),  # 加速左转
            8: (nx_min, 8, bank),  # 减速左转

            9: (0.5, 8, 0),  # 匀速爬升
            10: (nx_max, 8, 0),  # 加速爬升
            11: (nx_min, 8, 0),  # 减速爬升

            12: (0, 0.5, 0),  # 匀速俯冲
            13: (nx_max, 0.5, 0),  # 加速俯冲
            14: (nx_min, 0.5, 0),  # 减速俯冲
        }

        # 使用 get 获取动作，如果 id 不存在则返回默认值 (0, 1, 0)
        return actions.get(action_id, (0, 1, 0))


if __name__ == "__main__":
    model = AircraftModel()
    state = model.initiate()

    dt = 0.1  # 每步模拟 0.1 秒
    history = []
    print("开始模拟飞行...")
    cnt = 0

    for t in range(50):  # 模拟 10 秒 (100 * 0.1)
        # 1. 计算这一瞬间的变化率
        if t % 5 == 0:
            cnt += 1
            cnt = cnt % 14
        nx, ny, gamma = model.get_action(cnt)
        derivs = model.dynamics(t, state, nx, ny, gamma)  # 稍微给点推力和拉杆

        # 2. 【核心】状态更新：新状态 = 旧状态 + 变化率 * dt
        # 使用 numpy 的向量加法极其方便
        state = state + np.array(derivs) * dt

        history.append(state.copy())
        # 3. 每秒打印一次进度
        if t % 1 == 0:
            print(f"时间: {t * dt:.1f}s | 纬度距离: {state[0]:.2f}m | 经度距离: {state[2]:.2f}m | 高度: {state[1]:.2f}m | 速度: {state[3]:.2f}m/s")

    history = np.array(history)

    # 提取 x, y, z 坐标
    # 注意：你的代码中 y 是高度，x/z 是水平面
    x_pts = history[:, 0]
    y_pts = history[:, 1]  # 高度
    z_pts = history[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 轨迹
    ax.plot(x_pts, z_pts, y_pts, label='UAV Trajectory', color='b', lw=2)

    # 设置标签 (对应你的定义：x, z 是水平，y 是高度)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Altitude Y (m)')
    ax.set_title('3D Flight Path')

    # 优化视角：为了看清高度，我们可以设置起点的坐标
    ax.scatter(x_pts[0], z_pts[0], y_pts[0], color='green', label='Start')  # 起点
    ax.scatter(x_pts[-1], z_pts[-1], y_pts[-1], color='red', label='End')  # 终点

    plt.legend()
    plt.show()