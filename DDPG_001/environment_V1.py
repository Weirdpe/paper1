# from cmath import sin
import sys
from tkinter import N
import numpy as np

DEBUG = 1

np.random.seed(1)

# 状态：服务器队列状态、每个任务的优先级

# 地址:0-本地 1~n-服务器 n+1-云

# 服务器可以跨时隙，用户不能跨时隙计算，在本地卸载的任务都要在一个时隙内完成
# 有正在计算的任务用优先级 0 表示最高级


Task = np.dtype({"names": ['device_id', 'generate_slot', 'finish_slot', 'size', 'cpu', 'delay', 'priority'],
                 "formats": [np.int32, np.int32, np.float32, np.float32, np.float32, np.float32, np.int32]})

Action = np.dtype({"names": ["address"], "formats": [np.int32]})


class Configure():
    def __init__(self) -> None:
        # 每个基站用户的数量
        self.device_number = 20
        # 服务器数量
        self.server_number = 3
        # 每个服务器的设备数量
        self.device_set = np.array(
            (self.device_number, self.device_number, self.device_number))
        # 设备总数量
        self.device_sum_number = self.server_number*self.device_number
        # 用户的信道增益
        self.channel_gain = np.random.uniform(
            low=0.5, high=0.8, size=(self.device_sum_number))*1e-8
        # 噪声功率 e-13
        self.noise = np.random.uniform(
            low=0.7, high=0.9, size=(self.device_sum_number))*1e-13
        # 服务器之间的距离 m
        self.distance_server = np.random.randint(
            low=200, high=400, size=(self.server_number, self.server_number))
        # 服务器之间的传播速度 0.0001s/m
        self.trans_latency = 0.00005
        # 服务器CPU频率
        self.server_frequency = np.random.randint(
            low=10, high=15, size=(self.server_number))
        # CPU能量效率
        self.eta = 1e-28
        # 每bit需要的计算资源
        self.clock_per_bit = 600
        # 基站总带宽
        self.bandwidth_sum = 10
        # 每个用户分的基站带宽 1MHz
        self.bandwidth = self.bandwidth_sum / self.device_number
        # 服务器到云的速度 Mb/s
        self.cloud_rate = 1
        # 时隙长度 0.1s 100ms
        self.slot_length = 0.1
        # 服务器最大队列长度
        self.Q_server_max = 30
        # 本地设备最大队列长度
        self.Q_device_max = 5
        # 不同优先级的任务大小，每个任务都必须要在一个时隙内完成

        # 设置任务大小范围 Mb
        self.size_high_min = 0.01e6
        self.size_high_max = 0.1e6

        self.size_mid_min = 0.1e6
        self.size_mid_max = 0.2e6

        self.size_low_min = 0.2e6
        self.size_low_max = 0.3e6

        # 设置需要的时延范围
        self.delay_high_min = 0.2
        self.delay_high_max = 0.3

        self.delay_mid_min = 0.3
        self.delay_mid_max = 0.5

        self.delay_low_min = 0.5
        self.delay_low_max = 0.8

        # 云的传输速度 单位Mb / s
        self.cloud_transmisson_rate = 1

        # 超时惩罚
        self.overtime_penalty = 10

        # 设置随机数种子
        np.random.seed(1)


class Data():
    def __init__(self, max_slot, user_number) -> None:
        self.max_slot = max_slot
        self.task = np.zeros(shape=(self.max_slot, user_number), dtype=Task)
        # 计算能耗
        self.energy_compute = np.zeros(shape=(self.max_slot, user_number))
        # 传输能耗
        self.energy_transmission = np.zeros(shape=(self.max_slot, user_number))
        # 记录时延，单位用s
        self.delay = np.zeros(shape=(self.max_slot, user_number), dtype=float)
        # 丢弃的任务数量
        self.drop_task_number = np.zeros(shape=(self.max_slot))
        # 每个时隙设备总队列长度
        self.Q_device_size = np.zeros(self.max_slot)
        # 每个时隙服务器总队列长度
        self.Q_server_size = np.zeros(self.max_slot)

    def reset(self):
        # 将数据全部置为0
        self.task.fill(0)
        self.energy_compute.fill(0)
        self.energy_transmission.fill(0)
        self.delay.fill(0)
        self.drop_task_number.fill(0)
        self.Q_device_size.fill(0)
        self.Q_server_size.fill(0)
        pass


class Env():
    def __init__(self) -> None:
        # 参数设置
        self.parameter = Configure()
        self.data = Data(1001, self.parameter.device_sum_number)

        # 当前时隙
        self.slot = 0

        # 动作和网络维度
        self.action_dim = self.parameter.device_sum_number
        self.observation_dim = self.parameter.server_number*self.parameter.Q_server_max\
            + self.parameter.device_sum_number

        # 动作约束
        self.action_min = 0
        self.action_max = 1

        # 设备CPU频率约束
        self.device_frequency_min = 0
        self.device_frequency_max = 1.0

        # 队列中任务计算到一半未完成，不会出队列
        # 初始化用户队列
        # self.Q_device = [[]
        #                  for device in range(self.parameter.device_sum_number)]
        # 初始化服务器队列
        self.Q_server = [[] for server in range(self.parameter.server_number)]

        # 用于存放当前时隙生成的任务
        self.Buffer_task = np.zeros(
            shape=(self.parameter.device_sum_number), dtype=Task)

        # 记录时延
        self.task_delay = np.zeros(shape=(self.parameter.device_sum_number))
        # 计算消耗的能量
        self.energy_compute = np.zeros(self.parameter.device_sum_number)
        # 传输消耗的能量
        self.energy_transmission = np.zeros(self.parameter.device_sum_number)

        # 服务器时间保存点：保存未计算的任务在下一时隙消耗的时间（跨时隙计算？）
        self.points_server = np.zeros(self.parameter.server_number)

        self.drop = 0

    def step(self, action):
        self.drop = 0
        # 时间统一单位，按照时隙计算
        # 更新服务器队列
        # 处理的是上个时隙传输到服务器的任务
        for server in range(self.parameter.server_number):
            # 从上个任务时间开始
            time = self.points_server[server]
            if time > self.parameter.slot_length:
                self.points_server[server] = time - self.parameter.slot_length
                continue
            # 将上一时隙未完成的任务去掉队列
            if time > 0:
                self.Q_server[server].pop(0)
            # 取每个任务队列的头节点
            while len(self.Q_server[server]) > 0:
                task = self.Q_server[server][0]
                # 计算任务的处理时间 统一转化为s
                time = time + (task["cpu"] /
                               self.parameter.server_frequency[server])

                # 如果这个任务当前时隙无法计算完成，下一个时隙继续计算，任务不出队列
                if time > self.parameter.slot_length:
                    # 转化为时隙
                    finish_time = self.slot + time*10
                    task["finish_slot"] = finish_time
                    task["priority"] = 0
                    # 时延转化为S
                    delay = (task["finish_slot"] - task["generate_slot"])/10

                    # 如果超时，当作丢弃，加入到回报函数中
                    if delay > task["delay"]:
                        self.data.drop_task_number[self.slot] += 1
                    # 记录任务时延
                    self.data.delay[task["generate_slot"]
                                    ][task["device_id"]] = delay
                    # 更新数据数组中任务完成的时间
                    self.data.task[task["generate_slot"]
                                   ][task["device_id"]]["finish_slot"] = finish_time
                    # 记录任务断点
                    self.points_server[server] = time - \
                        self.parameter.slot_length

                    # self.Q_server[server].pop(0)
                    break
                else:
                    # 转化为时隙
                    finish_time = self.slot + time*10
                    task["finish_slot"] = finish_time
                    delay = (task["finish_slot"] - task["generate_slot"])/10

                    # 如果超时，当作丢弃，加入到回报函数中
                    if(delay > task["delay"]):
                        self.data.drop_task_number[self.slot] += 1

                    # 记录任务时延
                    self.data.delay[task["generate_slot"]
                                    ][task["device_id"]] = delay
                    # 用来替换任务数据
                    self.data.task[task["generate_slot"]
                                   ][task["device_id"]]["finish_slot"] = finish_time
                    self.Q_server[server].pop(0)
                    # 如果正好等于这个时隙长度 结束
                    if time == self.parameter.slot_length:
                        break
            # 如果该时隙没有跨时隙任务或者任务计算完毕，下一时隙时间从0开始
            if time <= self.parameter.slot_length:
                self.points_server[server] = 0

        if DEBUG:
            # print(self.Q_server)
            # print(self.__get_server_size())
            pass

        # 计算消耗的能量
        self.energy_compute = self.energy_compute * 0
        # 传输消耗的能量
        self.energy_transmission = self.energy_transmission * 0

        # 开始传输任务
        # 定义服务器任务缓存
        server_buffer = [[] for server in range(self.parameter.server_number)]
        for device in range(len(self.Buffer_task)):
            time = 0
            task = self.Buffer_task[device]
            size = task["size"]
            address = action[device]["address"]

            # 根据卸载位置，决定任务放置位置
            # 本地计算
            if address == 0:
                # 卸载到本地队列，单个时隙内计算完，根据任务求出CPU时钟周期
                frequency, finish = self.__get_cpu_frequency(task)
                delay = np.min([task["delay"], self.parameter.slot_length])
                self.energy_compute[device] = self.__energy_compute(
                    frequency, delay)
                # 判断一下是否可以计算完成这个任务
                if finish:
                    self.task_delay[device] = delay
                else:
                    # 这里加上时延惩罚
                    self.task_delay[device] = self.parameter.overtime_penalty
                    self.drop += 1

            # 卸载到边缘服务器
            # 服务迁移时延、无计算能耗
            elif address >= 1 and address <= self.parameter.server_number:
                # 考虑边缘服务器队列已经满了
                # 先卸载到边缘服务器，放到服务器任务缓存里面
                # 这里先求出传输能耗
                trans_time = self.parameter.slot_length
                trans_power = 0
                # 如果在覆盖范围内
                if self.__coverage(device, address-1):
                    trans_power = self.__transmisson_power(
                        device, size, trans_time)
                else:
                    origin_server = self.__get_server(device)
                    # 获得时间
                    trans_time = self.__get_time(origin_server, address-1)
                    trans_power = self.__transmisson_power(
                        device, size, trans_time)

                self.energy_transmission[device] = trans_power * trans_time
                if DEBUG:
                    # print("trans_time: ", trans_time)
                    # print("power: ", trans_power)
                    pass

                # self.data.energy_transmission[self.slot][device] = self.energy_transmission[device]
                # 将任务添加到对应的队列中
                # 需要对传输后的任务进行排序，先将任务放到对应的缓存里面，然后排序后添加到服务器任务队列
                server_buffer[address-1].append(task)

                # self.Q_server[address-1].append(task)

            # 卸载到云
            elif address == self.parameter.server_number+1:
                # 计算云的时延
                cloud_delay = size/self.parameter.cloud_rate
                delay = self.parameter.slot_length + cloud_delay

                # 如果超时，当作丢弃，加入到回报函数中
                if delay > task["delay"]:
                    # 任务超时，添加超时惩罚
                    self.task_delay[device] = self.parameter.overtime_penalty
                    self.drop += 1
                else:
                    self.task_delay[device] = delay
                # 记录任务时延
                # self.data.delay[task["generate_slot"]
                #                 ][task["device_id"]] = delay
                finish_time = self.slot + 1 + cloud_delay*10
                # self.data.task[task["generate_slot"]
                #                ][task["device_id"]]["finish_slot"] = finish_time

                # 计算传输能耗
                trans_time = self.parameter.slot_length
                trans_power = self.__transmisson_power(
                    device, size, trans_time)
                self.energy_transmission[device] = trans_power * trans_time
                # self.data.energy_transmission[self.slot][device] = self.energy_transmission[device]
        # 对传输到任务缓存里面的任务进行处理
        for server in range(self.parameter.server_number):
            # 对Buffer里面的任务进行排序
            server_buffer[server].sort(key=self.task_weight)
            insert_length = self.parameter.Q_server_max - \
                len(self.Q_server[server])
            # 根据队列的排序来计算时延
            delay = self.points_server[server] + self.parameter.slot_length
            for index in range(len(server_buffer[server])):
                task = server_buffer[server][index]
                if index < insert_length:
                    delay = delay + \
                        task["cpu"] / \
                        self.parameter.server_frequency[server]
                    if delay > task["delay"]:
                        # 任务超时的惩罚
                        self.task_delay[task["device_id"]
                                        ] = self.parameter.overtime_penalty
                        self.drop += 1
                    else:
                        self.task_delay[task["device_id"]] = delay
                else:
                    # 任务被丢弃的惩罚
                    self.task_delay[task["device_id"]
                                    ] = self.parameter.overtime_penalty
                    self.drop += 1

            # 如果超出最长的队列限制怎么办
            self.Q_server[server] = self.Q_server[server] + \
                server_buffer[server][:insert_length]

        # 传输完成，Buffer清空
        self.Buffer_task = np.zeros(
            shape=(self.parameter.device_sum_number), dtype=Task)

        reward, energy, delay = self.reward()
        self.slot = self.slot + 1

        # 开始生成任务，得到状态
        next_observation = self.observation()

        done = False
        drop_task = np.sum(self.task_delay == self.parameter.overtime_penalty)
        # 此时的服务器任务队列已经是该时隙卸载后的
        info = {"server size": self.__get_server_size(), "energy": energy,
                "delay": delay, "drop task": drop_task}
        return next_observation, reward, done, info

    def reset(self):
        # 重置数据数组
        self.data.reset()
        self.drop = 0
        # 初始化用户队列
        # self.Q_device = [[]
        #                  for device in range(self.parameter.device_sum_number)]
        # 初始化设备队列
        self.Q_server = [[] for server in range(self.parameter.server_number)]
        # 重置Buffer
        self.Buffer_task = np.zeros(
            shape=(self.parameter.device_sum_number), dtype=Task)
        # 重置服务器时间保存点
        self.points_server = np.zeros(self.parameter.server_number)
        # 重置时间
        self.slot = 0
        observation = self.observation()
        return observation

    # 生成任务，返回当前状态:设备任务队列 服务器任务队列 生成的任务
    def observation(self):
        # 服务器队列状态
        # 默认的优先级为4
        server_observation = np.zeros(
            self.parameter.server_number*self.parameter.Q_server_max) + 4
        # index = 0
        # 服务器队列中任务的优先级
        for server in range(self.parameter.server_number):
            index = server * self.parameter.Q_server_max
            for task in range(len(self.Q_server[server])):
                # 如果队列为空，则无 priority 属性
                server_observation[index] = self.task_weight(
                    self.Q_server[server][task])
                index = index + 1

        task_observation = []
        # 开始产生任务并得到任务优先级
        for index in range(self.parameter.device_sum_number):
            # 生成任务
            task = self.produce_task(index)
            # 保存已经生成的任务，在任务完成后将完成的时间填充
            self.data.task[self.slot][index] = task.copy()
            # 待传输任务
            self.Buffer_task[index] = task
            # self.Q_transmission[index].append(task)
            # task_observation.append(task["priority"])
            task_observation.append(self.task_weight(task))

        task_observation = np.array(task_observation)

        observation = np.concatenate(
            (server_observation, task_observation), axis=0)
        return observation

    # 开始计算reward
    def reward(self):
        # 根据队列长度、消耗的能量、队列内优先级方差计算
        # 队列长度使用大小表示
        # 加上时延，将超时的任务给出更大的惩罚

        Q_server_length = 0
        # Q_device_lenght = 0

        Q_server_size = 0
        # Q_device_size = 0
        for server in range(self.parameter.server_number):
            Q_server_length = Q_server_length + len(self.Q_server[server])
            for task in range(len(self.Q_server[server])):
                Q_server_size = Q_server_size + \
                    self.Q_server[server][task]["size"]

        # self.data.Q_server_size[self.slot] = Q_server_size
        # self.data.Q_device_size[self.slot] = Q_device_size

        # before_slot = np.clip(self.slot, 0, self.data.max_slot)

        # 任务丢弃数量
        # drop_task_number = self.data.drop_task_number[self.slot]
        # 计算能耗
        # energy_compute = np.sum(self.data.energy_compute[self.slot])
        # energy_transmission = np.sum(self.data.energy_transmission[self.slot])
        # energy = energy_compute + energy_transmission
        # reward = drop_task_number + energy

        delay = np.sum(self.task_delay)
        energy = np.sum(self.energy_compute) + np.sum(self.energy_transmission)
        if DEBUG:
            if self.slot == 500:
                # 查看超时的任务数量
                print("drop: ", self.drop)
                print("energy: ", energy)
                print("delay: ", delay)

        # reward = delay + energy * 100
        # delay ^ energy
        # reward = pow(delay, energy)
        reward = energy * 10 + \
            np.sum(self.task_delay == self.parameter.overtime_penalty) * \
            self.parameter.overtime_penalty
        return -reward, energy, delay

    # 计算传输功率
    # 传输功率公式：r = wlog(1+ph/n)
    # t = s / r
    # e = p*s/wlog(1+ph/n)
    # r/w = log(1+ph/n)
    # 1+ph/n = power(2, value)
    def __transmisson_power(self, device, size, time):
        # 将时隙长度转化为s
        rate = size/time
        value = rate/self.parameter.bandwidth
        value = np.power(2, value)
        value = value - 1
        value = value * self.parameter.noise[device]
        power = value/self.parameter.channel_gain[device]

        power = (np.power(2, rate/self.parameter.bandwidth) - 1) * \
            self.parameter.noise[device]/self.parameter.channel_gain[device]
        return power

    # 判断这个设备是否在该基站覆盖范围内
    def __coverage(self, device, server):
        # 按照每个基站没有固定的设备来算
        for ser in range(self.parameter.server_number):
            # 小于0就证明当前的服务器就是该设备所属的服务器
            if device - self.parameter.device_set[ser] < 0:
                # 得到对应的服务器ser
                if ser == server:
                    return True
                else:
                    return False
            device = device - self.parameter.device_set[ser]
        return False

    # 得到设备对应的是那个服务器
    def __get_server(self, device):
        # 按照每个基站没有固定的设备来算
        for ser in range(self.parameter.server_number):
            # 小于0就证明当前的服务器就是该设备所属的服务器
            if device - self.parameter.device_set[ser] < 0:
                # 得到对应的服务器ser
                return ser
            device = device - self.parameter.device_set[ser]
        return -1

    # 得到服务器之间的传输时延
    def __get_time(self, origin, target):
        # 按照距离计算时延
        time = self.parameter.trans_latency * \
            self.parameter.distance_server[origin][target]

        if DEBUG:
            # print("time: ", time)
            pass
        time = self.parameter.slot_length - time
        return time

    # 计算处理任务消耗的能量
    # 计算公式：eta * f^2 * c(计算的任务量) -> eta * f^3 * t
    # 时间单位是秒（s）
    def __energy_compute(self, frequency, time):
        frequency = frequency * 1e9
        energy = self.parameter.eta * frequency**3 * time
        return energy

    def __get_cpu_frequency(self, task):
        time = np.min([task["delay"], self.parameter.slot_length])
        frequency = task["cpu"]/time
        frequency = np.min([frequency, self.device_frequency_max])
        finish = True
        if time > self.parameter.slot_length:
            finish = False
        return frequency, finish

    # 产生任务
    def produce_task(self, device_id):
        # 生成任务的优先级
        priority = np.random.randint(low=1, high=4)
        size = 0
        cpu = 0
        delay = 0
        # 高优先级
        if priority == 1:
            size = np.random.randint(
                low=self.parameter.size_high_min, high=self.parameter.size_high_max)/1e6
            # 转换为GHz
            cpu = size * self.parameter.clock_per_bit / 1e3
            delay = np.random.uniform(
                low=self.parameter.delay_high_min, high=self.parameter.delay_high_max)
        elif priority == 2:
            size = np.random.randint(
                low=self.parameter.size_mid_min, high=self.parameter.size_mid_max)/1e6
            cpu = size * self.parameter.clock_per_bit / 1e3
            delay = np.random.uniform(
                low=self.parameter.delay_mid_min, high=self.parameter.delay_mid_max)
        elif priority == 3:
            size = np.random.randint(
                low=self.parameter.size_low_min, high=self.parameter.size_low_max)/1e6
            cpu = size * self.parameter.clock_per_bit / 1e3
            delay = np.random.uniform(
                low=self.parameter.delay_low_min, high=self.parameter.delay_low_max)

        task = np.array((device_id, self.slot, -1, size,
                        cpu, delay, priority), dtype=Task)
        return task

    # 计算传输速度
    def transmission_speed(self, device, power):
        sinr = np.log2(
            1+power*self.parameter.channel_gain[device]/self.parameter.noise[device])
        rate = self.parameter.bandwidth * sinr
        return rate

    # 种类越小，时延要求越低，排的越前
    def task_weight(self, elem):
        # 优先级的值：1，2，3
        # 时延的值：0.1 ~ 1.9
        # 将单位转化为S
        value = elem["priority"] + elem["delay"]
        return value

    # 任务采用泊松分布
    def __possion(self):
        pass

    def __get_server_size(self):
        Q_server_size = 0
        for server in range(self.parameter.server_number):
            for task in range(len(self.Q_server[server])):
                Q_server_size = Q_server_size + \
                    self.Q_server[server][task]["size"]
        return Q_server_size


if __name__ == "__main__":
    env = Env()
    print(env.Q_server)
    print(env.reset())
    actoin_dim = env.action_dim
    for step in range(10):
        action = np.random.randint(
            low=0, high=env.parameter.server_number + 2, size=actoin_dim)
        # action = action * 0
        action = np.array(action, dtype=Action)
        print(action)
        env.step(action)
