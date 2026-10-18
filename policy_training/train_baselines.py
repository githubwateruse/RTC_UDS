# -*- coding: utf-8 -*-
"""
Interactive prompt flow:
    1) synthetic_baseline | segmented_baseline
       -> synthetic  : 5min | 15min
       -> segmented  : 2000 | 2000-2004
            -> 2000       : 5min | 15min | 1h | 6h
            -> 2000-2004  : 5min | 15min
"""

import os
import time
import gc
from pathlib import Path

import numpy as np
import xlrd
import xlsxwriter
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages


DATA_ROOT = Path(os.environ.get(
    "ASTLINGEN_DATA_ROOT",
    Path(__file__).resolve().parents[1] / "SWMM_Astlingen",
))
OUTPUT_ROOT = Path(os.environ.get(
    "ASTLINGEN_OUTPUT_ROOT",
    Path(__file__).resolve().parent / "training_outputs",
))

MASTER_SYNTHETIC = DATA_ROOT / "synthetic_2000.inp"
MASTER_OBSERVED = DATA_ROOT / "observed_2000_2004.inp"
TMP_INP = DATA_ROOT / "_tmp_episode.inp"


# Hyperparameters for Evolution Strategies
N_KID = 10
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "600"))
LR = .05                   # learning rate
SIGMA = .05                # mutation strength / step size

# control time step in seconds; overwritten from the interactive prompt in __main__
CONTROL_TIME_STEP = 900

MONTH_LAST_DAY = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
STEP_SECONDS = {"5min": 300, "15min": 900, "1h": 3600, "6h": 21600}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Environmental programming module: SWMM simulation
class BasicEnv(object):
    def __init__(self, inp_file):
        self.input_file = os.path.normpath(inp_file)
        self.control_time_step = CONTROL_TIME_STEP  # control time step in seconds

        # initialize Simulation
        self.sim = Simulation(self.input_file)  # read input file
        self.sim.step_advance(self.control_time_step)  # set control time step

        # initialize Objects
        self._init_objects()

        self.sim.start()
        self.t = 1

        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds() / self.control_time_step)

        self.state = np.array([
            self.T1.depth, self.T2.depth, self.T3.depth,
            self.T4.depth, self.T5.depth, self.T6.depth,
        ])

    def _init_objects(self):
        # init node object, link object, and subcatchment object
        subcatchment_object = Subcatchments(self.sim)
        self.SC01 = subcatchment_object["SC01"]
        self.SC02 = subcatchment_object["SC02"]
        self.SC03 = subcatchment_object["SC03"]
        self.SC04 = subcatchment_object["SC04"]
        self.SC05 = subcatchment_object["SC05"]
        self.SC06 = subcatchment_object["SC06"]
        self.SC07 = subcatchment_object["SC07"]
        self.SC08 = subcatchment_object["SC08"]
        self.SC09 = subcatchment_object["SC09"]
        self.SC010 = subcatchment_object["SC010"]

        node_object = Nodes(self.sim)
        self.T1 = node_object["T1"]; self.T2 = node_object["T2"]; self.T3 = node_object["T3"]
        self.T4 = node_object["T4"]; self.T5 = node_object["T5"]; self.T6 = node_object["T6"]
        self.CSO7 = node_object["CSO7"]; self.CSO8 = node_object["CSO8"]
        self.CSO9 = node_object["CSO9"]; self.CSO10 = node_object["CSO10"]
        for i in range(1, 20):
            setattr(self, f"J{i}", node_object[f"J{i}"])
        self.Out_to_WWTP = node_object["Out_to_WWTP"]

        link_object = Links(self.sim)
        self.C14 = link_object["C14"]
        self.V1 = link_object["V1"]; self.V2 = link_object["V2"]; self.V3 = link_object["V3"]
        self.V4 = link_object["V4"]; self.V5 = link_object["V5"]; self.V6 = link_object["V6"]

        raingage_object = RainGages(self.sim)
        self.RG1 = raingage_object["RG1"]; self.RG2 = raingage_object["RG2"]
        self.RG3 = raingage_object["RG3"]; self.RG4 = raingage_object["RG4"]

        # the reward weights of the 6 storage units and 4 CSOs
        self.T1_weight, self.T2_weight, self.T3_weight, self.T4_weight, self.T5_weight, self.T6_weight = 1, 1, 1, 1, 1, 2
        self.CSO7_weight, self.CSO8_weight, self.CSO9_weight, self.CSO10_weight = 2, 1, 2, 1

        # the statistics of CSO volume of each node
        self.T1_CSO, self.T2_CSO, self.T3_CSO, self.T4_CSO, self.T5_CSO, self.T6_CSO = 0, 0, 0, 0, 0, 0
        self.CSO7_CSO, self.CSO8_CSO, self.CSO9_CSO, self.CSO10_CSO = 0, 0, 0, 0

    def step(self, action):
        # six orifices
        self.V1.target_setting = np.round(np.double(action[0]), decimals=2)
        self.V2.target_setting = np.round(np.double(action[1]), decimals=2)
        self.V3.target_setting = np.round(np.double(action[2]), decimals=2)
        self.V4.target_setting = np.round(np.double(action[3]), decimals=2)
        self.V5.target_setting = np.round(np.double(action[4]), decimals=2)
        self.V6.target_setting = np.round(np.double(action[5]), decimals=2)

        self.sim.__next__()

        self.state = np.array([
            self.T1.depth, self.T2.depth, self.T3.depth,
            self.T4.depth, self.T5.depth, self.T6.depth,
        ])

        Step_CSO_volume_T1 = self.T1.statistics['flooding_volume'] - self.T1_CSO
        Step_CSO_volume_T2 = self.T2.statistics['flooding_volume'] - self.T2_CSO
        Step_CSO_volume_T3 = self.T3.statistics['flooding_volume'] - self.T3_CSO
        Step_CSO_volume_T4 = self.T4.statistics['flooding_volume'] - self.T4_CSO
        Step_CSO_volume_T5 = self.T5.statistics['flooding_volume'] - self.T5_CSO
        Step_CSO_volume_T6 = self.T6.statistics['flooding_volume'] - self.T6_CSO

        Step_CSO_volume = self.T1_weight * Step_CSO_volume_T1 + self.T2_weight * Step_CSO_volume_T2  \
                      + self.T3_weight * Step_CSO_volume_T3 + self.T4_weight * Step_CSO_volume_T4  \
                      + self.T5_weight * Step_CSO_volume_T5 + self.T6_weight * Step_CSO_volume_T6  \
                      + self.CSO7_weight * (self.CSO7.statistics['flooding_volume'] - self.CSO7_CSO) + self.CSO8_weight * (self.CSO8.statistics['flooding_volume'] - self.CSO8_CSO) \
                      + self.CSO9_weight * (self.CSO9.statistics['flooding_volume'] - self.CSO9_CSO) + self.CSO10_weight * (self.CSO10.statistics['flooding_volume'] - self.CSO10_CSO)

        Reward_CSO_volume = -1 * Step_CSO_volume / 1000

        self.T1_CSO, self.T2_CSO = self.T1.statistics['flooding_volume'], self.T2.statistics['flooding_volume']
        self.T3_CSO, self.T4_CSO = self.T3.statistics['flooding_volume'], self.T4.statistics['flooding_volume']
        self.T5_CSO, self.T6_CSO = self.T5.statistics['flooding_volume'], self.T6.statistics['flooding_volume']

        self.CSO7_CSO, self.CSO8_CSO = self.CSO7.statistics['flooding_volume'], self.CSO8.statistics['flooding_volume']
        self.CSO9_CSO, self.CSO10_CSO = self.CSO9.statistics['flooding_volume'], self.CSO10.statistics['flooding_volume']

        if self.t < self.T - 1:
            done_steps = False
        else:
            done_steps = True

        self.t += 1

        return self.state, Reward_CSO_volume, done_steps, Step_CSO_volume_T1, Step_CSO_volume_T2, \
            Step_CSO_volume_T3, Step_CSO_volume_T4, Step_CSO_volume_T5, Step_CSO_volume_T6

    def reset(self):
        # initialize Simulation
        self.sim = Simulation(self.input_file)  # read input file
        self.sim.step_advance(self.control_time_step)  # set control time step

        # initialize Objects
        self._init_objects()

        self.sim.start()
        self.t = 1

        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds() / self.control_time_step)

        self.state = np.array([
            self.T1.depth, self.T2.depth, self.T3.depth,
            self.T4.depth, self.T5.depth, self.T6.depth,
        ])

        return self.state, self.T

    def close(self):
        self.sim.report()
        self.sim.close()
        del self.sim
        gc.collect()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data: runtime window slicing into a temporary .inp
def _observed_rain_path(year, month):

    def one(g):
        if month is None:
            name = f"{g}Astlingen_Erft{g}_2000.txt"
        elif year == 2000:
            name = f"{g}Astlingen_Erft{g}_{month}.txt"
        else:
            name = f"{g}Astlingen_Erft{g}_{year}_{month}.txt"
        return (DATA_ROOT / "rainfall_training" / "observed" / name).resolve().as_posix()
    return [one(1), one(2), one(3), one(4)]


def make_temp_inp(year=None, month=None, validation=False):

    if validation:
        start, end = "01/01/2000", "12/31/2000"
        rain = _observed_rain_path(2000, None)
    else:
        start = f"{month:02d}/01/{year}"                        # SWMM date format is MM/DD/YYYY
        end = f"{month:02d}/{MONTH_LAST_DAY[month]:02d}/{year}"
        rain = _observed_rain_path(year, month)

    out = []
    for line in MASTER_OBSERVED.read_text().splitlines():
        s = line.strip()
        if s.startswith("START_DATE"):
            line = "START_DATE           " + start
        elif s.startswith("REPORT_START_DATE"):
            line = "REPORT_START_DATE    " + start
        elif s.startswith("END_DATE"):
            line = "END_DATE             " + end
        elif s.startswith("rain1"):
            line = f'rain1            FILE "{rain[0]}"'
        elif s.startswith("rain2"):
            line = f'rain2            FILE "{rain[1]}"'
        elif s.startswith("rain3"):
            line = f'rain3            FILE "{rain[2]}"'
        elif s.startswith("rain4"):
            line = f'rain4            FILE "{rain[3]}"'
        out.append(line)

    TMP_INP.write_text("\n".join(out) + "\n")
    return str(TMP_INP)


def get_year_month(index):

    index = index % 60
    year = 2000 + index // 12
    month = index % 12 + 1
    return year, month


def episode_inp(num_generation, run_mode):

    if run_mode == "synthetic":
        return str(MASTER_SYNTHETIC)
    if run_mode == "segmented_2000":
        month = num_generation % 12 + 1
        return make_temp_inp(2000, month)
    # segmented_2000_2004
    year, month = get_year_month(num_generation)
    return make_temp_inp(year, month)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# some functions of Evolution Strategies
def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params, dtype=np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):  # reshape to be a matrix
    p, start = [], 0
    for shape in shapes:
        n_w, n_b = shape[0] * shape[1], shape[1]
        p.append(params[start: start + n_w].reshape(shape))
        p.append(params[start + n_w: start + n_w + n_b].reshape((1, shape[1])))
        start += n_w + n_b
    return p


def build_net(len_state, len_action, random_seed):
    rng = np.random.default_rng(random_seed)

    def linear(n_in, n_out):  # network linear layer
        w = rng.standard_normal(n_in * n_out, dtype=np.float32) * 0.1
        b = rng.standard_normal(n_out, dtype=np.float32) * 0.1
        return (n_in, n_out), np.concatenate((w, b))

    s0, p0 = linear(len_state, 30)
    s1, p1 = linear(30, 30)
    s2, p2 = linear(30, len_action)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


def get_action(params, x):
    x = x[np.newaxis, :]

    x = x.dot(params[0])
    x += params[1]
    np.maximum(x, 0, out=x)

    x = x.dot(params[2])
    x += params[3]
    np.maximum(x, 0, out=x)

    x = x.dot(params[4])
    x += params[5]
    return 0.5 * np.tanh(x)[0] + 0.5  # continuous action


def get_reward_per_event(swmm_file, network_params):
    ep_r = 0
    storm_env = BasicEnv(swmm_file)
    Storm_state, EP_LEN = storm_env.reset()

    for t in range(EP_LEN):
        Orifice_setting = get_action(network_params, Storm_state)
        New_Storm_state, R, done_steps, CSO_T1, CSO_T2, CSO_T3, CSO_T4, CSO_T5, CSO_T6 = storm_env.step(Orifice_setting)

        ep_r = ep_r + R
        Storm_state = New_Storm_state

        if float(done_steps):
            storm_env.close()
            break

    return ep_r


def get_reward_training_events(swmm_file, shapes, params, seed_and_id=None):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        rng = np.random.default_rng(seed)
        params += sign(k_id) * SIGMA * rng.standard_normal(params.size, dtype=np.float32)

    network_params = params_reshape(shapes, params)
    EP_R = get_reward_per_event(swmm_file, network_params)

    return EP_R


# Train
# --------------------------------------------------------------------------------
def train(net_shapes, net_params, optimizer, utility, num_generation, run_mode):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.arange(num_generation * N_KID + 1, num_generation * N_KID + N_KID + 1).repeat(2)  # mirrored sampling

    # slice the episode window ONCE per generation; all kids reuse the same file
    swmm_file = episode_inp(num_generation, run_mode)

    rewards = np.empty(2 * N_KID, dtype=np.float32)
    for k_id in range(N_KID * 2):
        reward = get_reward_training_events(swmm_file, net_shapes, net_params, [noise_seed[k_id], k_id])
        rewards[k_id] = reward

    """ update with reward rank and utility """
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = np.zeros_like(net_params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        rng = np.random.default_rng(noise_seed[k_id])   # reconstruct noise using seed
        cumulative_update += utility[ui] * sign(k_id) * rng.standard_normal(net_params.size, dtype=np.float32)

    gradients = optimizer.get_gradients(cumulative_update / (2 * N_KID * SIGMA))
    return net_params + gradients


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Interactive configuration
def _choose(header, options):
    """options: list of (key, label). Returns the chosen key."""
    print(header)
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i}) {label}")
    while True:
        ans = input("enter number: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1][0]
        print("  invalid input, please enter a number from the list.")


def prompt_config():
    mode = _choose("Step 1: choose baseline",
                   [("synthetic", "synthetic_baseline"),
                    ("segmented", "segmented_baseline")])

    if mode == "synthetic":
        dt = _choose("Step 2: choose step size",
                     [("5min", "5min"), ("15min", "15min")])
        run_mode = "synthetic"
    else:
        period = _choose("Step 2: choose period",
                         [("2000", "2000"), ("2000-2004", "2000-2004")])
        if period == "2000":
            dt = _choose("Step 3: choose step size",
                         [("5min", "5min"), ("15min", "15min"),
                          ("1h", "1h"), ("6h", "6h")])
            run_mode = "segmented_2000"
        else:
            dt = _choose("Step 3: choose step size",
                         [("5min", "5min"), ("15min", "15min")])
            run_mode = "segmented_2000_2004"

    return run_mode, STEP_SECONDS[dt], dt


def result_folder(run_mode, dt_label):
    if run_mode == "synthetic":
        return f"neuro_evolution_baseline_synthetic_{dt_label}"
    if run_mode == "segmented_2000":
        return f"neuro_evolution_baseline_segmented_2000_{dt_label}"
    return f"neuro_evolution_baseline_segmented_2000_2004_{dt_label}"


if __name__ == "__main__":
    run_mode, CONTROL_TIME_STEP, dt_label = prompt_config()

    result_path = str(OUTPUT_ROOT / result_folder(run_mode, dt_label)) + "/"
    os.makedirs(result_path, exist_ok=True)

    print(f"\nrun_mode={run_mode} | control_time_step={CONTROL_TIME_STEP}s | "
          f"output={result_path}\n")

    Train_time_start = time.time()

    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)

    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    LEN_STATE = 6
    LEN_ACTION = 6
    # the structure of each neural network is identical, so their shapes are the same
    net_shapes, net_params = build_net(LEN_STATE, LEN_ACTION, 0)
    optimizer = SGD(net_params, LR)

    START_GENERATION = int(os.environ.get("START_GENERATION", "0"))
    if START_GENERATION != 0:
        path_0 = result_path + "/Gen_" + str(START_GENERATION - 1) + ".xlsx"
        workbook = xlrd.open_workbook(path_0)
        data_sheet = workbook.sheet_by_index(0)
        net_params = np.array(data_sheet.col_values(0))

    for g in range(NUM_GENERATIONS):
        t0 = time.time()

        num_generation = g + START_GENERATION
        net_params = train(net_shapes, net_params, optimizer, utility, num_generation, run_mode)

        # test trained net without noise (validated on observed year 2000)
        valid_file = make_temp_inp(validation=True)
        net_r = get_reward_training_events(valid_file, net_shapes, net_params, seed_and_id=None)

        print(
            '| Gen: ', num_generation,
            '| Net_R: %.3f' % net_r,
            '| Gen_T: %.2f' % (time.time() - t0),)

        # record the parameters in each generation
        training_time = time.time() - Train_time_start
        workbook_params = xlsxwriter.Workbook(result_path + 'Gen_' + str(num_generation) + '.xlsx')
        worksheet_params = workbook_params.add_worksheet('sheet1')
        worksheet_params.write_column(0, 0, net_params)
        worksheet_params.write_column(0, 1, [net_r])
        worksheet_params.write_column(0, 2, [training_time])
        workbook_params.close()

        del t0, num_generation, net_r, workbook_params, worksheet_params, valid_file
        gc.collect()

    Train_time_end = time.time()
    print("total time consumption", Train_time_end - Train_time_start)
