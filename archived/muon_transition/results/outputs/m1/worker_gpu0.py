
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/b205/projects/training_dynamics_lab/muon_transition_experiment")
sys.path.insert(0, "/home/b205/projects/training_dynamics_lab/muon_transition_experiment/src")
from scripts.m1_run import cell_worker
cell_worker(json.loads('"[{\\"run_id\\": \\"m1_muon_lr0.5_wd0.01_p0.5_s2030\\", \\"stage\\": \\"M1\\", \\"arm\\": \\"muon\\", \\"p\\": 0.5, \\"seed\\": 2030, \\"lr_multiplier\\": 0.5, \\"lr\\": 0.0025, \\"reference_lr\\": 0.005, \\"weight_decay\\": 0.01}, {\\"run_id\\": \\"m1_muon_lr0.5_wd0.02_p0.5_s2030\\", \\"stage\\": \\"M1\\", \\"arm\\": \\"muon\\", \\"p\\": 0.5, \\"seed\\": 2030, \\"lr_multiplier\\": 0.5, \\"lr\\": 0.0025, \\"reference_lr\\": 0.005, \\"weight_decay\\": 0.02}, {\\"run_id\\": \\"m1_muon_lr1.0_wd0.01_p0.5_s2030\\", \\"stage\\": \\"M1\\", \\"arm\\": \\"muon\\", \\"p\\": 0.5, \\"seed\\": 2030, \\"lr_multiplier\\": 1.0, \\"lr\\": 0.005, \\"reference_lr\\": 0.005, \\"weight_decay\\": 0.01}]"'), "0")
