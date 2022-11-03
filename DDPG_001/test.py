import numpy as np
import math

from env import Vehicle, RSU


tasks_category = [
    {
    'task_size': 20,
    'amount_freq': 30,
    'max_latency': 0.2,
    'task_prior': 3,
    'step_index': 2,
    'is_done': False},
    {'task_size': 30,
     'amount_freq': 20,
     'max_latency': 0.4,
     'task_prior': 2,
     'step_index': 4,
     'is_done': False},
    {'task_size': 40,
     'amount_freq': 40,
     'max_latency': 0.3,
     'task_prior': 1,
     'step_index': 3,
     'is_done': False}
]

vecs = [
    {
        'vec_id':2,
        'loc_v_x': 1,
        'loc_v_y': 1,
        'speed': 20,
        'compute_freq': 20,
        'queue_size': 100,
        'working_size': 20,
        'cache_size': 30,
        'task': tasks_category},
    {
        'vec_id':3,
        'loc_v_x': 2,
        'loc_v_y': 3,
        'speed': 30,
        'compute_freq': 20,
        'queue_size': 100,
        'working_size': 20,
        'cache_size': 30,
        'task': tasks_category}
]

vehicles = []
vehicles.insert(2, Vehicle(
        vecs[1]['vec_id'],
        vecs[1]['loc_v_x'],
        vecs[1]['loc_v_y'],
        vecs[1]['speed'],
        vecs[1]['compute_freq'],
        vecs[1]['queue_size'],
        vecs[1]['working_size'],
        vecs[1]['cache_size'],
        vecs[1]['task'],
))
# for i in range(2):
#     vehicles.append(Vehicle(
#         vecs[i]['vec_id'],
#         vecs[i]['loc_v_x'],
#         vecs[i]['loc_v_y'],
#         vecs[i]['speed'],
#         vecs[i]['compute_freq'],
#         vecs[i]['queue_size'],
#         vecs[i]['working_size'],
#         vecs[i]['cache_size'],
#         vecs[i]['task'],
#     )
# )
print(vehicles[0].vec_id)
V_number = 8
R_number = 10
V_R_range = np.zeros(shape=(V_number, R_number), dtype=int)

V_R_range[2, 7] = 1

for i in range(R_number):
    if V_R_range[2, i] == 1:
        print(i)
rsu_x = np.linspace(0, 1000, num=R_number)
# print(rsu_x)
vec_loc = [2, 4]
rsu_loc = [4, 7]
dis = math.sqrt(math.pow((rsu_loc[0] - vec_loc[0]), 2) + math.pow((rsu_loc[1] - vec_loc[1]), 2))
# print(dis)



