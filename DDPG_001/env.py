import math

import scipy.io as sio
import numpy as np

import math as mh

"""
奖励函数
This is a update
"""

"""
计算时延：
任务量/计算频率；  任务量=系数c * 任务数据大小

传输时延：
任务数据大小/传输速率

等待时间

汽车移动模型

能耗模型

存储容量

定义RSU的位置：Average RSU's transmission range:600m
Inter-RSU distance(D):2-6km
Vehicle density: 0.003 - 0.007vehicle/m
"""
"""
状态空间设计：
任务大小 计算资源大小 最大容忍时延 平均队列剩余空间大小  
"""

"""
车辆类：
位置信息：loc_v_x, loc_v_y
速度:speed
CPU频率:compute_v_freq
队列大小:queue_v_size
计算中的队列大小:working_v_size
等待队列大小：cache_v_size
剩余队列大小 = queue_v_size - working_v_size - cache_v_size
"""

v_y_random = [5, 15, 25, 35]
print(v_y_random[np.random.randint(4)])

speed_random = [3, 5, 7]
compute_freq_random = [4, 7]
compute_r_freq_random = [30, 50, 60]
queue_size_random = [100, 200, 300]
queue_r_size_random = [1000, 2000, 3000, 4000]
vecs = []

Task = np.dtype({"names": ['vec_id', 'generate_slot', 'finish_slot', 'size', 'cpu', 'delay', 'priority'],
                 "formats": [np.int32, np.int32, np.float32, np.float32, np.float32, np.float32, np.int32]})

# 判断在哪个RSU的范围内
"""
任务： 
任务大小 task_size
需要多少周期 amountFreq
容忍时间 max_latency
任务优先级 task_prior 
哪个时隙产生的：step_index
是否做完:is_done
"""

tasks_category = [
    {
        'task_size': 20,
        'amount_freq': 30,
        'max_latency': 0.2,
        'task_prior': 3,
        'step_index': 2,
        'is_done': False
    },
    {
        'task_size': 30,
        'amount_freq': 20,
        'max_latency': 0.4,
        'task_prior': 2,
        'step_index': 4,
        'is_done': False
    },
    {
        'task_size': 40,
        'amount_freq': 40,
        'max_latency': 0.3,
        'task_prior': 1,
        'step_index': 3,
        'is_done': False
    }
]
"""
车辆数组：可由for循环往数组中加入随机生成的数据
"""
# vecs = [
#     {
#         'loc_v_x': 1,
#         'loc_v_y': 1,
#         'speed': 20,
#         'compute_freq': 20,
#         'queue_size': 100,
#         'working_size': 20,
#         'cache_size': 30,
#         'task': tasks_category
#     },
#     {
#         'loc_v_x': 2,
#         'loc_v_y': 3,
#         'speed': 30,
#         'compute_freq': 20,
#         'queue_size': 100,
#         'working_size': 20,
#         'cache_size': 30,
#         'task': tasks_category
#     }
# ]


v_range_rsu = []  # 判断车辆属于哪个RSU

"""
有m种任务，车辆每次从这些任务中随机抽取k个，可以重复。

"""

# vehicle1 = Vehicle(1, 2, 20, 30, 100, 50, 60, tasks_category)

"""
RSU类：
位置信息：loc_r_x, loc_r_y
CPU频率：compute_r_freq
队列大小：queue_r_size
计算中的队列大小:working_r_size
等待队列大小：cache_r_size
剩余队列大小: = queue_r_size - working_r_size - cache_r_size
"""


class Vehicle(object):
    def __init__(self, vec_id, loc_v_x, loc_v_y, speed, compute_freq, queue_size, task):
        self.vec_id = vec_id
        self.loc_v_x = loc_v_x
        self.loc_v_y = loc_v_y
        self.vec_loc = [loc_v_x, loc_v_y]
        self.speed = speed
        self.compute_freq = compute_freq
        self.queue_size = queue_size
        self.used_size = 0
        self.remaining_size = queue_size - self.used_size
        self.task = task  # 字典列表保存任务
        self.task_queue = []


class RSU(object):
    def __init__(self, loc_r_x, loc_r_y, compute_r_freq, queue_size):
        self.loc_r_x = loc_r_x
        self.loc_r_y = loc_r_y
        self.rsu_loc = [loc_r_x, loc_r_y]
        self.compute_r_freq = compute_r_freq
        self.queue_size = queue_size
        self.used_size = 0
        self.remaining_size = self.queue_size - self.used_size
        self.task_queue = []


class MBS(object):
    def __init__(self, loc_m_x, loc_m_y, compute_r_freq, queue_size):
        self.loc_m_x = loc_m_x
        self.loc_m_y = loc_m_y
        self.mbs_loc = [loc_m_x, loc_m_y]
        self.compute_r_freq = compute_r_freq
        self.queue_size = queue_size
        self.used_size = 0
        self.remaining_size = self.queue_size - self.used_size
        self.task_queue = []


class IIov_Env(object):
    def __init__(self, max_slot, vehicle_number, rsu_number):
        self.vehicle_number = vehicle_number
        self.rsu_number = rsu_number
        self.vehicles = []
        self.vehicle_info = []
        self.rsus = []
        self.rsu_info = []
        self.mbs_info = []
        self.max_slot = max_slot
        self.task = np.zeros(shape=(self.max_slot, self.vehicle_number + self.rsu_number + 1), dtype=Task)
        # 'loc_m_x': 1,
        # 'loc_m_y': 2,
        # 'compute_r_freq': 2,
        # 'queue_size': 2
        self.v_r_range = np.zeros(shape=(vehicle_number, rsu_number), dtype=int)
        self.MBS = None

        # 当前时隙
        self.slot = 0
        # 每bit需要的计算资源
        self.clock_per_bit = 600

        # 车辆服务器数量
        self.server_number_vec = 20

        # RSU服务器数量
        self.server_number_rsu = 50

        # 设置任务大小
        self.size_high_min = 0.01e6
        self.size_high_max = 0.1e6

        self.size_mid_min = 0.1e6
        self.size_mid_max = 0.2e6

        self.size_low_min = 0.2e6
        self.size_low_max = 0.3e6

        # 设置时延范围
        self.delay_high_min = 0.2
        self.delay_high_max = 0.3

        self.delay_mid_min = 0.3
        self.delay_mid_max = 0.5

        self.delay_low_min = 0.5
        self.delay_low_max = 0.8

        # 优先级最高为3，最低为1
        self.priority_max = 3
        self.priority_min = 1

        # 车辆任务队列
        self.vec_queue = [[] for _ in range(self.vehicle_number)]
        # RSU任务队列
        self.rsu_queue = [[] for _ in range(self.rsu_number)]
        # MBS任务队列
        self.mbs_queue = []
        # 用户的信道增益
        self.channel_gain = np.random.uniform(
            low=0.5, high=0.8, size=self.vehicle_number) * 1e-8
        # 噪声功率 e-13
        self.noise = np.random.uniform(low=0.7, high=0.9, size=self.vehicle_number) * 1e-13
        # 服务器CPU频率
        self.vec_frequency = np.random.randint(low=2, high=6, size=self.vehicle_number)
        # 用于存放该时隙下生成的任务
        self.current_production = np.zeros(shape=self.vehicle_number, dtype=Task)

        # 基站总带宽
        self.bandwidth_sum = 10
        # 每个用户分的基站带宽
        self.bandwidth = self.bandwidth_sum / self.vehicle_number
        # 时隙长度 0.1s 100ms
        self.slot_length = 0.1
        # 车辆队列长度
        self.vec_queue_max = 30
        # RSU队列长度
        self.rsu_queue_max = 50
        # MBS队列长度
        self.mbs_queue_max = 80

        # 计算能耗时的系数 mu
        self.mu = 1e-27

        # log函数底
        self.a = 0.6

        # 任务权重 a1和a2
        self.a1 = 0.8
        self.a2 = 0.5

        # 用于存放当前时隙生成的任务
        self.Buffer_task = np.zeros(shape=self.vehicle_number, dtype=Task)
        # 记录时延
        self.task_delay = np.zeros(shape=self.vehicle_number)
        # 计算消耗的能量
        self.energy_compute = np.zeros(shape=self.vehicle_number)
        # 传输消耗的能量
        self.energy_transmission = np.zeros(shape=self.vehicle_number)
        # 设置随机数种子
        np.random.seed(1)

    # 注入信息
    def _generate_architecture_info(self):
        # vehicle
        for i in range(self.vehicle_number):
            self.vehicle_info.append({
                'vec_id': i + 1,
                'loc_v_x': 4,
                'loc_v_y': v_y_random[np.random.randint(4)],
                'speed': speed_random[np.random.randint(3)],
                'compute_freq': compute_freq_random[np.random.randint(2)],
                'queue_size': queue_size_random[np.random.randint(3)],
                'task': tasks_category

            })
        # rsu
        rsu_x = np.linspace(0, 100, num=self.rsu_number)
        for i in range(self.rsu_number):
            self.rsu_info.append({
                'loc_r_x': rsu_x[i],
                'loc_r_y': 40,
                'compute_r_freq': compute_r_freq_random[np.random.randint(3)],
                'queue_r_size': queue_r_size_random[np.random.randint(4)],
                'working_r_size': 0,
                'cache_r_size': 0
            })
        # mbs
        self.mbs_info.append({
            'loc_m_x': 1,
            'loc_m_y': 2,
            'compute_r_freq': 2,
            'queue_size': 2
        })

    # 生成对象
    def _generate_architecture(self):
        # vehicle
        for vehicle_i in range(self.vehicle_number):
            self.vehicles.append(Vehicle(
                self.vehicle_info[vehicle_i]['vec_id'],
                self.vehicle_info[vehicle_i]['loc_v_x'],
                self.vehicle_info[vehicle_i]['loc_v_y'],
                self.vehicle_info[vehicle_i]['speed'],
                self.vehicle_info[vehicle_i]['compute_freq'],
                self.vehicle_info[vehicle_i]['queue_size'],
                self.vehicle_info[vehicle_i]['task']
            ))
        # rsu
        for rsu_i in range(self.rsu_number):
            self.rsus.append(RSU(
                self.rsu_info[rsu_i]['loc_r_x'],
                self.rsu_info[rsu_i]['loc_r_y'],
                self.rsu_info[rsu_i]['compute_r_freq'],
                self.rsu_info[rsu_i]['queue_r_size']
            ))

        self.MBS = MBS(self.mbs_info[0]['loc_m_x'],
                       self.mbs_info[0]['loc_m_y'],
                       self.mbs_info[0]['compute_r_freq'],
                       self.mbs_info[0]['queue_size']
                       )

    # 判断车辆属于哪个RSU范围内，修改相应的数组
    def v_r_judgement(self):
        for vec_i in range(self.vehicle_number):
            for rsu_i in range(self.rsu_number):
                if math.sqrt(math.pow((self.vehicles[vec_i].vec_loc[0] - self.rsus[rsu_i].rsu_loc[0]), 2) + \
                             math.pow((self.vehicles[vec_i].vec_loc[0] - self.rsus[rsu_i].rsu_loc[0]), 2)) <= 20:
                    self.v_r_range[vec_i, rsu_i] = 1

    # 生成任务，返回当前状态
    def observation(self):
        # 车辆状态
        vehicle_queue_observation = np.zeros(self.vehicle_number)
        # RSU状态
        rsu_queue_observation = np.zeros(self.rsu_number)
        # MBS状态
        mbs_queue_observation = []

        offloadisornot = np.ones(shape=(self.vehicle_number + self.rsu_number + 1))
        remaining_size = []  # 剩余空间数组

        # 计算各队列中任务的总大小和剩余空间大小
        # vehicles
        for vec_i in range(self.vehicle_number):
            length1 = len(self.vehicles[vec_i].task_queue)
            if length1 > 0:
                for i in range(length1):
                    self.vehicles[vec_i].used_size += self.vehicles[vec_i].task_queue[i].size
                remaining_size.append(self.vehicles[vec_i].remaining_size)
                offloadisornot[vec_i] = 0

        # RSU
        for rsu_i in range(self.vehicle_number + 1, self.vehicle_number + self.rsu_number + 1):
            length2 = len(self.rsus[rsu_i].task_queue)
            if length2 > 0:
                for i in range(length2):
                    self.rsus[rsu_i].used_size += self.rsus[rsu_i].task_queue[i].size
                remaining_size.append(self.rsus[rsu_i].remainging_size)
                offloadisornot[rsu_i] = 0
        # MBS
        length3 = len(self.MBS.task_queue)
        if length3 > 0:
            for i in range(length3):
                self.MBS.used_size += self.MBS.task_queue[i].size
            remaining_size.append(self.MBS.remaining_size)
            offloadisornot[self.vehicle_number + self.rsu_number] = 0

        task_observation = []
        # 开始产生任务并得到任务优先级
        for index in range(self.vehicle_number):
            # 生成任务
            task = self.generate_task(index)
            self.task[self.slot][index] = task.copy()
            self.Buffer_task[index] = task

            task_observation.append(self.take_weight(task['priority'], task['delay']))

        task_observation = np.array(task_observation)

        return vehicle_queue_observation, rsu_queue_observation, mbs_queue_observation, task_observation

    def step(self, action):
        # 
        obs = 0
        reward = 0
        done = False
        extra_info = None
        return obs, reward, done, extra_info

    def reward(self):
        pass

    def reset(self):
        pass

    def generate_task(self, vec_id):
        priority = np.random.randint(low=1, high=4)
        size = 0
        cpu = 0
        delay = 0
        # 高优先级
        if priority == 1:
            size = np.random.randint(
                low=self.size_high_min, high=self.size_high_max) / 1e6
            # 转换为GHZ
            cpu = size * self.clock_per_bit / 1e3
            delay = np.random.uniform(low=self.delay_high_min, high=self.delay_high_max)
        elif priority == 2:
            size = np.random.randint(
                low=self.size_mid_min, high=self.size_mid_max) / 1e6
            # 转换为GHZ
            cpu = size * self.clock_per_bit / 1e3
            delay = np.random.uniform(low=self.delay_mid_min, high=self.delay_mid_max)
        elif priority == 3:
            size = np.random.randint(
                low=self.size_low_min, high=self.size_low_max) / 1e6
            # 转换为GHZ
            cpu = size * self.clock_per_bit / 1e3
            delay = np.random.uniform(low=self.delay_low_min, high=self.delay_low_max)

        task = np.array((vec_id, self.slot, -1, size,
                         cpu, delay, priority), dtype=Task)
        return task

    # 归一化函数
    def nl(self, x):
        nor = 0
        if 1 <= x <= 3:
            nor = (x - self.priority_min) / (self.priority_max - self.priority_min)
        elif 0.2 <= x <= 0.3:
            nor = (x - self.delay_high_min) / (self.delay_high_max - self.delay_high_min)
        elif 0.3 <= x <= 0.5:
            nor = (x - self.delay_mid_min) / (self.delay_mid_max - self.delay_mid_min)
        elif 0.5 <= x <= 0.8:
            nor = (x - self.delay_low_min) / (self.delay_low_max - self.delay_low_min)
        return nor

    def take_weight(self, priority, delay):
        return self.log(self.a1 * self.nl(priority) + self.a2 * self.nl(delay))

    def transmission_speed(self, vec_id, power):
        sinr = np.log2(
            1 + power * self.channel_gain[vec_id] / self.noise[vec_id])
        rate = self.bandwidth * sinr
        return rate

    # 计算处理任务消耗的能量
    # e = mu * omega * f ^ 2
    # mu = 1e-27
    def __energy_compute(self, frequency, workload):
        energy = self.mu * workload * frequency ** 2
        return energy

    # 计算传输任务消耗的能量
    # e = p * T(传输)
    def __energy_transmit(self, power, trans_time):
        energy = power * trans_time
        return energy

    def __get_cpu_frequency(self):
        pass

    def __time_compute(self, workload, frequency):
        time = workload / frequency
        return time

    def __time_transmit(self, task_size, rate):
        return task_size / rate

    # log函数
    def log(self, x):
        return np.log(x) / np.log(self.a)


env = IIov_Env(8, 9, 1001)
# print(env.v_r_range)
env._generate_architecture_info()
env._generate_architecture()
# env.v_r_judgement()
print(env.nl(0.78))
print(env.take_weight(2, 0.38))
# print(env.observation())
# print("车辆的id号为", env.vehicles[3].vec_id)

# print(env.rsu_queue)
# print(env.v_r_range)
# print(env.rsus[1].rsu_loc,env.rsus[2].rsu_loc)
