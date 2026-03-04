import jsbsim
from jsbsim import FGFDMExec
import gymnasium as gym

class FlightEnv(gym.Env):
    def __init__(self):
        self.fdm = FGFDMExec('jsbsim-master')
        self.fdm.load_model('f16')
        self.fdm.set_debug_level(0)

        self.reset()

    def reset(self):
        self.fdm["ic/h-sl-ft"] = 9000.0 / 0.3048
        self.fdm['ic/lat-geod-deg'] = 60.0
        self.fdm["ic/psi-true-deg"] = 180.0
        self.fdm['ic/long-gc-deg'] = 120.0
        self.fdm['ic/vt-fps'] = 300.0 / 0.3048
        self.fdm.run_ic()

        self.fdm["fcs/left-aileron-cmd-norm"] = 0.0
        self.fdm["fcs/right-aileron-cmd-norm"] = 0.0
        self.fdm["fcs/elevator-cmd-norm"] = 0.0
        self.fdm["fcs/rudder-cmd-norm"] = 0.0
        self.fdm["fcs/throttle-cmd-norm"] = 0.8  # 维持平飞的推力

    def get_observation(self):
        '''经纬高'''
        altitude = self.fdm['position/h-sl-ft'] #ft
        latitude = self.fdm['position/lat-geod-rad']
        longitude = self.fdm['position/long-gc-rad']

        '''俯仰角theta 滚转角phi 航向角psi'''
        theta = self.fdm['attitude/theta-rad']
        phi = self.fdm['attitude/phi-rad']
        psi = self.fdm['attitude/psi-rad']

        '''攻角（过大过小导致失速） 最大/最小攻角 侧滑角'''
        alpha = self.fdm['aero/alpha-rad']
        alpha_max = self.fdm['aero/alpha-max-rad']
        alpha_min = self.fdm['aero/alpha-min-rad']
        beta = self.fdm['aero/beta-rad']

        '''舵面值 速度值'''
        aileron_right = self.fdm['fcs/right-aileron-pos-norm']
        aileron_left = self.fdm['fcs/left-aileron-pos-norm']
        elevator = self.fdm['fcs/elevator-pos-norm']
        rudder = self.fdm['fcs/rudder-pos-norm']
        v = self.fdm['velocities/vt-fps']

        '''机体系速度分量u前 v右 w下'''
        v_u = self.fdm['velocities/u-fps']
        v_v = self.fdm['velocities/v-fps']
        v_w = self.fdm['velocities/w-fps']

        '''NED系速度分量'''
        v_north = self.fdm['velocities/v-north-fps']
        v_east = self.fdm['velocities/v-east-fps']
        v_down = self.fdm['velocities/v-down-fps']

        '''俯仰q/滚转p/偏航r角速度'''
        roll_rate = self.fdm['velocities/p-rad_sec']
        pitch_rate = self.fdm['velocities/q-rad_sec']
        yaw_rate = self.fdm['velocities/r-rad_sec']
