import numpy as np
import time
import math
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class FlightTrackEnv(gym.Env):
    def __init__(self):
        """
        初始化环境
        :param jsbsim_path: JSBSim 配置文件路径
        :param model_path: 练好的 LSTM 动力学模型路径
        """
        self.g = 9.81  # 重力加速度 (m/s^2)
        self.x = 0 #维度位置(m)
        self.y = 5000 #海拔(m)
        self.z = 0 #经度位置(m)
        self.v = 400 #(m/s)
        self.theta = np.deg2rad(0)
        self.psi = np.deg2rad(0)

        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dv = 0
        self.dtheta = 0
        self.dpsi = 0

        self.nx = 0
        self.ny = 1
        self.gamma = 0

        self.target_x = 0
        self.target_y = 0
        self.target_z = 0

        self.current_step = 0
        self.dt = 0.02
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(
            low=-np.inf,  # 下界：如果没有严格限制，填负无穷
            high=np.inf,  # 上界：填正无穷
            shape=(15,),  # 形状：14维向量
            dtype=np.float32  # 数据类型：通常用 float32 匹配神经网络
        )

        self.reset()

    def get_observation(self):

        obs = np.array([self.x, self.y, self.z,
                        self.v, self.theta, self.psi,
                        self.dx, self.dy, self.dz,
                        self.dv, self.dtheta, self.dpsi,
                        self.target_x, self.target_y, self.target_z])
        return obs

    def step(self, action_id):
        """核心控制循环：执行一步跟踪"""
        self._update_target()
        self.nx, self.ny, self.gamma = self.get_action(action_id)

        # 1. 运动学方程 (公式 1)
        self.dx = self.v * np.cos(self.theta) * np.cos(self.psi)
        self.dy = self.v * np.sin(self.theta)
        self.dz = -self.v * np.cos(self.theta) * np.sin(self.psi)

        # 2. 动力学方程 (公式 2 修正版)
        # v_dot: 速度变化率
        self.dv = self.g * (self.nx - np.sin(self.theta))
        self.dtheta = self.g / max(self.v, 0.1) * (self.ny * np.cos(self.gamma) - np.cos(self.theta))

        # psi_dot: 航向角变化率
        if abs(np.cos(self.theta)) < 1e-4:
            self.dpsi = 0
        else:
            self.dpsi = -self.g / (max(self.v, 0.1) * np.cos(self.theta)) * self.ny * np.sin(self.gamma)
        # gamma_dot: 爬升角变化率 (注意分母应为 v)

        self.x += self.dx * self.dt
        self.y += self.dy * self.dt
        self.z += self.dz * self.dt
        self.v += self.dv * self.dt
        self.theta += self.dtheta * self.dt
        self.psi += self.dpsi * self.dt

        obs = self.get_observation()
        reward = self._calculate_reward()
        self.current_step += 1
        truncated = False
        terminated = False
        if self.current_step >= 1000:
            truncated = True

        return obs, reward, truncated, terminated, {}

    def get_action(self, action_id):
        # 预计算常数
        bank = np.arccos(1 / 8)
        nx_max = 1.2
        nx_min = -0.8
        ny_max = 9.0
        ny_min = -3.0

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

    def _calculate_reward(self):
        """计算当前位置与目标的偏差"""
        # TODO: 欧氏距离或其他度量
        # 计算偏差向量 (Error vector)

        # 惩罚项：偏差的平方和 (即 ||e||^2)
        # 距离越远，惩罚呈指数级增长
        error_x = self.target_x - self.x
        error_y = self.target_y - self.y
        error_z = self.target_z - self.z
        error = np.array([error_x / 200, error_y / 200, error_z / 200])
        reward = -np.sum(np.square(error))
        return reward

    def _update_target(self):

        if self.current_step % 25 == 0:
            self.target_x += np.random.uniform(-100, 100)
            self.target_y += np.random.uniform(-100, 100)
            self.target_z += np.random.uniform(-100, 100)

    def reset(self):
        """重置环境到初始状态"""
        self.x = 0  # 维度位置(m)
        self.y = 5000  # 海拔(m)
        self.z = 0  # 经度位置(m)
        self.v = 400  # (m/s)
        self.theta = np.deg2rad(0)
        self.psi = np.deg2rad(0)

        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dv = 0
        self.dtheta = 0
        self.dpsi = 0

        self.target_x = self.x + np.random.uniform(-100, 100)
        self.target_y = self.y + np.random.uniform(-100, 100)
        self.target_z = self.z + np.random.uniform(-100, 100)

        obs = np.array([self.x, self.y, self.z,
                        self.v, self.theta, self.psi,
                        self.dx, self.dy, self.dz,
                        self.dv, self.dtheta, self.dpsi,
                        self.target_x, self.target_y, self.target_z])
        return obs

# 使用示例
if __name__ == "__main__":
    env = FlightTrackEnv()
    env.reset()
    history = []
    T = 10.0
    total_steps = int(math.ceil(T / env.dt))
    for _ in range(total_steps):
        action = env.action_space.sample()
        obs, reward, truncated, terminated, _ = env.step(action)
        history.append(obs)
        if truncated or terminated:
            print(f"结束")
            break
    history = np.array(history)
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
