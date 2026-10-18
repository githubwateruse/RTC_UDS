# -*- coding: utf-8 -*-
import numpy as np
import time
import os
import gc
import xlrd
import xlsxwriter
from pathlib import Path
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages

REPO = Path(__file__).resolve().parents[1]
SWMM_DIR = REPO / "SWMM_Astlingen"
BASE_INP = SWMM_DIR / "testing.inp"
TMP_INP = SWMM_DIR / "_tmp_test.inp"
TMP_RPT = SWMM_DIR / "_tmp_test.rpt"
TMP_OUT = SWMM_DIR / "_tmp_test.out"
POLICY_DIR = REPO / "trained_policies"
OUT_DIR = Path(__file__).resolve().parent / "testing_outputs"

DT_SECONDS = {"5min": 300, "15min": 900, "1h": 3600, "6h": 21600}

# (policy file, control interval, test start, test end, label)
POLICIES = [
    ("policy_synthetic_baseline_5min.xlsx", "5min", "01/01/2001", "12/31/2009", "Synthetic"),
    ("policy_synthetic_baseline_15min.xlsx", "15min", "01/01/2001", "12/31/2009", "Synthetic"),
    ("policy_segmented_baseline_2000_5min.xlsx", "5min", "01/01/2001", "12/31/2009", "Segmented"),
    ("policy_segmented_baseline_2000_15min.xlsx", "15min", "01/01/2001", "12/31/2009", "Segmented"),
    ("policy_segmented_baseline_2000_1h.xlsx", "1h", "01/01/2006", "12/31/2006", "Segmented_2006"),
    ("policy_segmented_baseline_2000_1h.xlsx", "1h", "01/01/2008", "12/31/2008", "Segmented_2008"),
    ("policy_segmented_baseline_2000_6h.xlsx", "6h", "01/01/2006", "12/31/2006", "Segmented_2006"),
    ("policy_segmented_baseline_2000_6h.xlsx", "6h", "01/01/2008", "12/31/2008", "Segmented_2008"),
    ("policy_segmented_baseline_2000_2004_5min.xlsx", "5min", "01/01/2005", "12/31/2009", "Segmented_2000_2004"),
    ("policy_segmented_baseline_2000_2004_15min.xlsx", "15min", "01/01/2005", "12/31/2009", "Segmented_2000_2004"),
    ("policy_segmented_continue_learning_2000_5min.xlsx", "5min", "01/01/2001", "12/31/2009", "Segmented_Continue_Learning"),
    ("policy_segmented_continue_learning_2000_15min.xlsx", "15min", "01/01/2001", "12/31/2009", "Segmented_Continue_Learning"),
    ("policy_segmented_continue_learning_2000_2004_5min.xlsx", "5min", "01/01/2005", "12/31/2009", "Segmented_Continue_Learning_2000_2004"),
    ("policy_segmented_continue_learning_2000_2004_15min.xlsx", "15min", "01/01/2005", "12/31/2009", "Segmented_Continue_Learning_2000_2004"),
]

# menu groups: (label, indices into POLICIES)
TEST_GROUPS = [
    ("synthetic + segmented, 2000, 5/15 min", [0, 1, 2, 3]),
    ("segmented, 2000-2004, 5/15 min", [8, 9]),
    ("continual learning, 2000, 5/15 min", [10, 11]),
    ("continual learning, 2000-2004, 5/15 min", [12, 13]),
    ("segmented, 2000, 1h/6h", [4, 5, 6, 7]),
    ("all", list(range(len(POLICIES)))),
]


def _choose(header, options):
    print(header)
    for i, (label, _) in enumerate(options, 1):
        print(f"  {i}) {label}")
    while True:
        ans = input("enter number: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        print("  invalid input, please enter a number from the list.")


def prompt_config():
    options = [(f"{label:<44}[{len(indices)} runs]", indices)
               for label, indices in TEST_GROUPS]
    _, indices = _choose("Step 1: choose test set", options)
    return [POLICIES[i] for i in indices]


def make_temp_test_inp(start, end):
    out = []
    for line in BASE_INP.read_text().splitlines():
        s = line.strip()
        if s.startswith("START_DATE"):
            line = "START_DATE           " + start
        elif s.startswith("END_DATE"):
            line = "END_DATE             " + end
        out.append(line)
    TMP_INP.write_text("\n".join(out) + "\n")
    return str(TMP_INP)


class BasicEnv(object):
    def __init__(self, inp_file, control_time_step):
        self.input_file = inp_file
        self.control_time_step = control_time_step

        self.sim = Simulation(self.input_file)
        self.sim.step_advance(self.control_time_step)

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

    def step(self, action):
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

        if self.t < self.T - 1:
            done_steps = False
        else:
            done_steps = True
        self.t += 1

        return self.state, done_steps

    def close(self):
        self.sim.report()
        self.sim.close()
        del self.sim
        gc.collect()


def params_reshape(shapes, params):
    p, start = [], 0
    for shape in shapes:
        n_w, n_b = shape[0] * shape[1], shape[1]
        p.append(params[start: start + n_w].reshape(shape))
        p.append(params[start + n_w: start + n_w + n_b].reshape((1, shape[1])))
        start += n_w + n_b
    return p


def build_net(len_state, len_action, random_seed):
    rng = np.random.default_rng(random_seed)

    def linear(n_in, n_out):
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
    return 0.5 * np.tanh(x)[0] + 0.5


def parse_rpt(rpt_path):
    lines = open(rpt_path, "r", -1, "utf-8").readlines()
    start_lines = end_lines = None
    for m in range(len(lines)):
        if "Node Flooding Summary" in lines[m]:
            start_lines = m + 10
        if "Storage Volume Summary" in lines[m]:
            end_lines = m - 4

    BBB, BBB_creek, BBB_river = [], [], []
    if start_lines is None or end_lines is None:
        return BBB, BBB_creek, BBB_river

    for n in range(start_lines, end_lines + 1):
        Res = lines[n].split('    ')
        cso_volume = np.round(float(Res[-2]), decimals=3)
        BBB.append(cso_volume)
        if Res[0] == '  CSO7' or Res[0] == '  CSO9' or Res[0] == '  T6':
            BBB_creek.append(cso_volume)
        else:
            BBB_river.append(cso_volume)
    return BBB, BBB_creek, BBB_river


if __name__ == "__main__":
    selected = prompt_config()
    OUT_DIR.mkdir(exist_ok=True)

    LEN_STATE = 6
    LEN_ACTION = 6
    net_shapes, _ = build_net(LEN_STATE, LEN_ACTION, 0)
    len_params = 6 * 30 + 30 + 30 * 30 + 30 + 30 * 6 + 6

    for fname, dt, start, end, label in selected:
        t0 = time.time()
        print(f"===== {dt}_Testing performance_{label} ({fname}) =====")

        inp = make_temp_test_inp(start, end)

        workbook = xlrd.open_workbook(str(POLICY_DIR / fname))
        data_sheet = workbook.sheet_by_index(0)
        net_params = np.array(data_sheet.col_values(0)[0:len_params])
        network_params = params_reshape(net_shapes, net_params)

        env = BasicEnv(inp, DT_SECONDS[dt])
        state = env.state
        for _ in range(env.T):
            state, done_steps = env.step(get_action(network_params, state))
            if done_steps:
                env.close()
                break

        BBB, BBB_creek, BBB_river = parse_rpt(str(TMP_RPT))
        Weighted_CSO_VOLUME = np.sum(BBB) + np.sum(BBB_creek)

        print("total CSO volume", np.sum(BBB))
        print("creek", np.sum(BBB_creek))
        print("river", np.sum(BBB_river))

        out_xlsx = OUT_DIR / f"{dt}_Testing performance_{label}.xlsx"
        workbook_curve = xlsxwriter.Workbook(str(out_xlsx))
        worksheet_curve = workbook_curve.add_worksheet('sheet1')
        worksheet_curve.write_column(0, 0, [Weighted_CSO_VOLUME])
        worksheet_curve.write_column(0, 1, [np.sum(BBB)])
        worksheet_curve.write_column(0, 2, [np.sum(BBB_creek)])
        worksheet_curve.write_column(0, 3, [np.sum(BBB_river)])
        workbook_curve.close()

        for p in (TMP_INP, TMP_RPT, TMP_OUT):
            if p.exists():
                os.remove(p)

        print("total time consumption", time.time() - t0)
