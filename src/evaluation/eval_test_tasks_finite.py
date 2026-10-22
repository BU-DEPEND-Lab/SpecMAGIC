import sys
sys.path.insert(0, 'src')

import argparse
import os

import pandas as pd
from tqdm import tqdm

# from evaluation.simulate import simulate
import multiprocessing as mp

import random

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env_safety
from ltl import FixedSampler
from model.model import build_model_safety
from model.agent import MultiAgent
from config import model_configs
from sequence.search import ExhaustiveSearchSafety, SpotBFS
from utils.model_store import ModelStore
import argparse



env_to_tasks = {
    "SafetyPointMultiGoal0-v0":{
        "ma3_ind": 
        [
            {
                "agent_0": '(!magenta_0 U (blue_0 | green_0)) & (F magenta_0)', # phi_1
                "agent_1": '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))', # phi_2
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F green_2)', # phi_3
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F green_0)', # phi_4
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F magenta_1)', # phi_5
                "agent_2": '(F magenta_2) & (!yellow_2 U (blue_2 | green_2))', # phi_6
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (blue_0 | magenta_0))', # phi_7
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F yellow_1)', # phi_8
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F blue_2)', # phi_9
            },
        ],
        "ma3_ind_OOD": 
        [
            {
                "agent_0": '!magenta_0 U (blue_0 & F (!yellow_0 U (green_0 & F magenta_0)))',
                "agent_1": '!blue_1 U (green_1 & F (!magenta_1 U (yellow_1 & F blue_1)))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F (!blue_2 U (green_2 & F magenta_2)))',
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F (!blue_0 U (green_0 & F magenta_0)))',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F (!green_1 U (magenta_1 & F yellow_1)))',
                "agent_2": '!yellow_2 U ((blue_2 | green_2) & F (!magenta_2 U (yellow_2 & F blue_2)))',
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (!(yellow_0 | magenta_0) U (blue_0 & F magenta_0)))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F (!yellow_1 U (magenta_1 & F green_1)))',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F (!magenta_2 U (blue_2 & F yellow_2)))',
            },
        ],
        "ma3_ind_req": 
        [
            {
                "agent_0": '(!magenta_0 U (blue_0 | green_0)) & (F magenta_0)',
                "agent_1": '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F green_2)',
                "req": "(blue_0 Req green_1) & (blue_1 Req green_2) & (blue_2 Req magenta_0)",
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F green_0)',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F magenta_1)',
                "agent_2": '(F magenta_2) & (!yellow_2 U (blue_2 | green_2))',
                "req": "(green_0 Req magenta_1) & (magenta_1 Req blue_2) & (magenta_2 Req green_1)",
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (blue_0 | magenta_0))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F yellow_1)',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F blue_2)',
                "req": "(blue_0 Req green_1) & (yellow_1 Req magenta_2) & (blue_2 Req yellow_0)",
            },
        ],
        "ma3_ind_req_OOD": 
        [
            {
                "agent_0": '!magenta_0 U (blue_0 & F (!yellow_0 U (green_0 & F magenta_0)))',
                "agent_1": '!blue_1 U (green_1 & F (!magenta_1 U (yellow_1 & F blue_1)))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F (!blue_2 U (green_2 & F magenta_2)))',
                "req": "(blue_0 Req green_1) & (blue_0 Req blue_2) & (green_1 Req green_0) & (blue_2 Req yellow_1)",
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F (!blue_0 U (green_0 & F magenta_0)))',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F (!green_1 U (magenta_1 & F yellow_1)))',
                "agent_2": '!yellow_2 U ((blue_2 | green_2) & F (!magenta_2 U (yellow_2 & F blue_2)))',
                "req": "(green_1 Req green_0) & (green_1 Req yellow_2) & (green_0 Req magenta_1) & (yellow_2 Req magenta_0)",
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (!(yellow_0 | magenta_0) U (blue_0 & F magenta_0)))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F (!yellow_1 U (magenta_1 & F green_1)))',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F (!magenta_2 U (blue_2 & F yellow_2)))',
                "req": "(yellow_0 Req magenta_2) & (yellow_0 Req magenta_1) & (magenta_2 Req blue_0) & (magenta_1 Req blue_2)",
            },
        ],
        "ma3_ind_cnt": 
        [
            {
                "agent_0": '(!magenta_0 U (blue_0 | green_0)) & (F magenta_0)',
                "agent_1": '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F green_2)',
                "cnt": "Sync[{0, 2}, yellow, 2] & Async[{0, 1, 2}, green, 3]",
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F green_0)',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F magenta_1)',
                "agent_2": '(F magenta_2) & (!yellow_2 U (blue_2 | green_2))',
                "cnt": "Sync[{0, 1, 2}, magenta, 2] & Async[{0, 2}, yellow, 2]",
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (blue_0 | magenta_0))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F yellow_1)',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F blue_2)',
                "cnt": "Sync[{1, 2}, green, 2] & Async[{0, 1, 2}, blue, 2]",
            },
        ],
        "ma3_ind_cnt_OOD": 
        [
            {
                "agent_0": '!magenta_0 U (blue_0 & F (!yellow_0 U (green_0 & F magenta_0)))',
                "agent_1": '!blue_1 U (green_1 & F (!magenta_1 U (yellow_1 & F blue_1)))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F (!blue_2 U (green_2 & F magenta_2)))',
                "cnt": "Sync[{0, 1, 2}, green, 2] & Async[{0, 1}, yellow, 1] & Async[{1, 2}, magenta, 1]",
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F (!blue_0 U (green_0 & F magenta_0)))',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F (!green_1 U (magenta_1 & F yellow_1)))',
                "agent_2": '!yellow_2 U ((blue_2 | green_2) & F (!magenta_2 U (yellow_2 & F blue_2)))',
                "cnt": "Async[{0, 1, 2}, magenta, 3] & Sync[{0, 2}, yellow, 1] & Sync[{0, 1, 2}, blue, 2]",
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (!(yellow_0 | magenta_0) U (blue_0 & F magenta_0)))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F (!yellow_1 U (magenta_1 & F green_1)))',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F (!magenta_2 U (blue_2 & F yellow_2)))',
                "cnt": "Sync[{1, 2}, green, 1] & Async[{0, 1, 2}, blue, 2] & Sync[{0, 1}, magenta, 1]",
            },
        ],
        "ma3_ind_req_cnt": 
        [
            {
                # automaton num_states = 83, num_edges = 1362
                "agent_0": '(!magenta_0 U (blue_0 | green_0)) & (F magenta_0)',
                "agent_1": '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F green_2)',
                "req": "(blue_0 Req green_1) & (blue_1 Req green_2) & (blue_2 Req magenta_0)",
                "cnt": "Sync[{0, 2}, yellow, 2] & Async[{0, 1, 2}, green, 3]",

                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 6, num_edges = 16
                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 9, num_edges = 35
                # DEBUG: automaton num_states = 16, num_edges = 81

            },
            {
                # automaton num_states = 148, num_edges = 3124 when total_spec = "&".join(specs)
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F green_0)',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F magenta_1)',
                "agent_2": '(F magenta_2) & (!yellow_2 U (blue_2 | green_2))',
                "req": "(green_0 Req magenta_1) & (magenta_1 Req blue_2) & (magenta_2 Req green_1)",
                "cnt": "Sync[{0, 1, 2}, magenta, 2] & Async[{0, 2}, yellow, 2]",

                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 7, num_edges = 24
                # DEBUG: automaton num_states = 8, num_edges = 27
            },
            {
                # automaton num_states = 68, num_edges = 1067
                "agent_0": '!green_0 U (yellow_0 & F (blue_0 | magenta_0))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F yellow_1)',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F blue_2)',
                "req": "(blue_0 Req green_1) & (yellow_1 Req magenta_2) & (blue_2 Req yellow_0)",
                "cnt": "Sync[{1, 2}, green, 2] & Async[{0, 1, 2}, blue, 2]",

                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 4, num_edges = 8
                # DEBUG: automaton num_states = 9, num_edges = 35
                # DEBUG: automaton num_states = 10, num_edges = 36
            },
        ],

        "ma3_req_n3": 
        [
            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'req': '(green_2 Req blue_0) & (blue_1 Req blue_0) & (blue_1 Req green_2)'
            },
            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'req': '(magenta_0 Req yellow_2) & (yellow_1 Req magenta_0) & (yellow_1 Req yellow_2)'
            },
            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'req': '(green_2 Req magenta_1) & (yellow_0 Req magenta_1) & (green_2 Req yellow_0)'
            },
        ],

        "ma3_req_n6":
        [
            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'req': '(yellow_1 Req yellow_0) & (blue_1 Req green_2) & (yellow_2 Req green_1) & (blue_2 Req green_1) & (blue_1 Req magenta_0) & (green_2 Req blue_0)'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'req': '(magenta_2 Req blue_1) & (magenta_0 Req yellow_1) & (magenta_2 Req green_0) & (yellow_2 Req magenta_0) & (yellow_0 Req yellow_2) & (yellow_2 Req magenta_1)'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'req': '(magenta_1 Req yellow_0) & (yellow_0 Req green_2) & (blue_0 Req magenta_1) & (magenta_0 Req green_1) & (green_0 Req green_1) & (green_0 Req green_2)'
            },
        ],


        "ma3_req_n9":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'req': '(green_2 Req blue_1) & (green_0 Req blue_1) & (blue_0 Req blue_2) & (yellow_2 Req yellow_1) & (yellow_2 Req magenta_0) & (green_2 Req yellow_0) & (green_2 Req green_0) & (blue_0 Req green_2) & (blue_0 Req green_1)'
            },
            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'req': '(magenta_0 Req yellow_2) & (yellow_0 Req magenta_1) & (magenta_0 Req magenta_2) & (magenta_0 Req blue_1) & (green_0 Req magenta_2) & (blue_0 Req magenta_1) & (yellow_1 Req yellow_2) & (magenta_0 Req yellow_1) & (green_0 Req green_1)'
            },
            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'req': '(green_1 Req magenta_0) & (green_0 Req yellow_2) & (green_0 Req blue_2) & (magenta_1 Req green_2) & (blue_0 Req magenta_2) & (green_0 Req green_2) & (green_2 Req yellow_1) & (green_2 Req yellow_0) & (green_1 Req yellow_0)'
            },

        ],

        "ma3_req_n12":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'req': '(blue_2 Req yellow_0) & (blue_2 Req green_1) & (blue_2 Req yellow_1) & (blue_2 Req magenta_1) & (blue_0 Req green_2) & (green_2 Req yellow_1) & (blue_1 Req blue_0) & (yellow_2 Req green_0) & (magenta_1 Req green_2) & (blue_1 Req magenta_0) & (magenta_0 Req magenta_1) & (yellow_2 Req yellow_0)'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'req': '(green_1 Req magenta_2) & (yellow_1 Req magenta_0) & (magenta_0 Req yellow_2) & (green_1 Req magenta_0) & (green_1 Req blue_2) & (yellow_0 Req yellow_2) & (magenta_1 Req magenta_2) & (green_0 Req yellow_2) & (green_2 Req magenta_0) & (green_2 Req yellow_0) & (blue_2 Req magenta_1) & (magenta_0 Req magenta_1)'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'req': '(green_0 Req blue_1) & (magenta_2 Req yellow_1) & (green_0 Req magenta_2) & (blue_0 Req yellow_1) & (blue_2 Req blue_1) & (green_2 Req yellow_1) & (green_2 Req blue_1) & (green_2 Req yellow_0) & (magenta_1 Req blue_0) & (magenta_1 Req green_2) & (magenta_0 Req magenta_2) & (magenta_1 Req magenta_0)'
            },

        ],

        "ma3_req_n15":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'req': '(yellow_2 Req blue_1) & (magenta_2 Req yellow_1) & (green_0 Req yellow_1) & (blue_2 Req blue_1) & (yellow_2 Req yellow_1) & (yellow_2 Req magenta_1) & (green_0 Req magenta_2) & (yellow_0 Req yellow_2) & (green_1 Req yellow_2) & (magenta_0 Req magenta_1) & (blue_0 Req blue_1) & (magenta_2 Req magenta_0) & (green_0 Req blue_1) & (blue_0 Req green_1) & (green_2 Req blue_0)'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'req': '(blue_2 Req magenta_1) & (green_1 Req green_0) & (yellow_2 Req magenta_1) & (magenta_2 Req magenta_1) & (yellow_2 Req magenta_0) & (blue_0 Req green_2) & (magenta_0 Req yellow_1) & (magenta_1 Req magenta_0) & (yellow_0 Req green_2) & (magenta_2 Req yellow_1) & (blue_2 Req blue_0) & (magenta_1 Req blue_0) & (yellow_1 Req green_2) & (blue_1 Req green_0) & (green_2 Req green_0)'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'req': '(green_2 Req magenta_1) & (magenta_1 Req green_0) & (yellow_0 Req green_1) & (green_2 Req yellow_1) & (yellow_0 Req magenta_1) & (green_1 Req green_0) & (green_2 Req blue_1) & (yellow_0 Req blue_2) & (blue_0 Req magenta_2) & (magenta_1 Req blue_2) & (green_2 Req yellow_0) & (magenta_0 Req magenta_2) & (green_2 Req green_1) & (green_1 Req yellow_2) & (yellow_2 Req blue_1)'
            },

        ],

        "ma3_cnt_n2":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'cnt': 'Sync[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, blue, 2]'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'cnt': 'Async[{0, 1, 2}, blue, 1] & Sync[{0, 1, 2}, blue, 1]'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'cnt': 'Async[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, blue, 2]'
            },

        ],

        "ma3_cnt_n4":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'cnt': 'Sync[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, green, 1] & Sync[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, blue, 3]'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'cnt': 'Sync[{0, 1, 2}, yellow, 2] & Async[{0, 1, 2}, yellow, 2] & Sync[{0, 1, 2}, yellow, 2] & Async[{0, 1, 2}, green, 2]'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'cnt': 'Sync[{0, 1, 2}, yellow, 3] & Async[{0, 1, 2}, green, 3] & Async[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, magenta, 2]'
            },

        ],

        "ma3_cnt_n6":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'cnt': 'Async[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, blue, 1] & Async[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, magenta, 3] & Async[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, yellow, 3]'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'cnt': 'Async[{0, 1, 2}, magenta, 2] & Sync[{0, 1, 2}, blue, 1] & Async[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, yellow, 3] & Sync[{0, 1, 2}, green, 1] & Sync[{0, 1, 2}, blue, 3]'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'cnt': 'Async[{0, 1, 2}, magenta, 2] & Async[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, yellow, 2] & Sync[{0, 1, 2}, blue, 2] & Sync[{0, 1, 2}, blue, 2] & Async[{0, 1, 2}, blue, 2]'
            },

        ],

        "ma3_cnt_n8":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'cnt': 'Sync[{0, 1, 2}, magenta, 2] & Sync[{0, 1, 2}, yellow, 3] & Sync[{0, 1, 2}, yellow, 1] & Async[{0, 1, 2}, green, 3] & Sync[{0, 1, 2}, green, 1] & Async[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, magenta, 1] & Sync[{0, 1, 2}, magenta, 2]'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'cnt': 'Sync[{0, 1, 2}, yellow, 1] & Async[{0, 1, 2}, yellow, 3] & Async[{0, 1, 2}, magenta, 2] & Async[{0, 1, 2}, blue, 2] & Sync[{0, 1, 2}, green, 2] & Async[{0, 1, 2}, magenta, 2] & Sync[{0, 1, 2}, magenta, 1] & Async[{0, 1, 2}, yellow, 3]'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'cnt': 'Async[{0, 1, 2}, green, 1] & Sync[{0, 1, 2}, blue, 1] & Sync[{0, 1, 2}, yellow, 2] & Async[{0, 1, 2}, yellow, 3] & Async[{0, 1, 2}, yellow, 3] & Sync[{0, 1, 2}, blue, 3] & Async[{0, 1, 2}, blue, 1] & Async[{0, 1, 2}, yellow, 1]'
            },

        ],
        "ma3_cnt_n10":
        [

            {
                'agent_0': 'F blue_0',
                'agent_1': 'F blue_1',
                'agent_2': 'F green_2',
                'cnt': 'Sync[{0, 1, 2}, magenta, 1] & Sync[{0, 1, 2}, magenta, 2] & Sync[{0, 1, 2}, blue, 3] & Async[{0, 1, 2}, yellow, 1] & Sync[{0, 1, 2}, yellow, 3] & Async[{0, 1, 2}, yellow, 3] & Sync[{0, 1, 2}, blue, 1] & Async[{0, 1, 2}, green, 1] & Async[{0, 1, 2}, yellow, 2] & Sync[{0, 1, 2}, blue, 2]'
            },

            {
                'agent_0': 'F magenta_0',
                'agent_1': 'F yellow_1',
                'agent_2': 'F yellow_2',
                'cnt': 'Async[{0, 1, 2}, green, 2] & Async[{0, 1, 2}, magenta, 1] & Sync[{0, 1, 2}, yellow, 2] & Sync[{0, 1, 2}, magenta, 2] & Sync[{0, 1, 2}, magenta, 3] & Sync[{0, 1, 2}, blue, 3] & Async[{0, 1, 2}, blue, 3] & Sync[{0, 1, 2}, green, 3] & Sync[{0, 1, 2}, green, 1] & Sync[{0, 1, 2}, yellow, 3]'
            },

            {
                'agent_0': 'F yellow_0',
                'agent_1': 'F magenta_1',
                'agent_2': 'F green_2',
                'cnt': 'Sync[{0, 1, 2}, magenta, 2] & Async[{0, 1, 2}, magenta, 2] & Async[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, blue, 1] & Sync[{0, 1, 2}, magenta, 1] & Async[{0, 1, 2}, green, 2] & Sync[{0, 1, 2}, yellow, 2] & Sync[{0, 1, 2}, blue, 1] & Sync[{0, 1, 2}, green, 2] & Async[{0, 1, 2}, magenta, 3]'
            },

        ],
        "ma3_ind_req_cnt_OOD": 
        [
            {
                "agent_0": '!magenta_0 U (blue_0 & F (!yellow_0 U (green_0 & F magenta_0)))',
                "agent_1": '!blue_1 U (green_1 & F (!magenta_1 U (yellow_1 & F blue_1)))',
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F (!blue_2 U (green_2 & F magenta_2)))',
                "req": "(blue_0 Req green_1) & (blue_0 Req blue_2) & (green_1 Req green_0) & (blue_2 Req yellow_1)",
                "cnt": "Sync[{0, 1, 2}, green, 2] & Async[{0, 1}, yellow, 1] & Async[{1, 2}, magenta, 1]",

                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 10, num_edges = 45
                # DEBUG: automaton num_states = 20, num_edges = 108
            },
            {
                "agent_0": '!magenta_0 U ((blue_0 | yellow_0) & F (!blue_0 U (green_0 & F magenta_0)))',
                "agent_1": '!(blue_1 | yellow_1) U (green_1 & F (!green_1 U (magenta_1 & F yellow_1)))',
                "agent_2": '!yellow_2 U ((blue_2 | green_2) & F (!magenta_2 U (yellow_2 & F blue_2)))',
                "req": "(green_1 Req green_0) & (green_1 Req yellow_2) & (green_0 Req magenta_1) & (yellow_2 Req magenta_0)",
                "cnt": "Async[{0, 1, 2}, magenta, 3] & Sync[{0, 2}, yellow, 1] & Sync[{0, 1, 2}, blue, 2]",

                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 10, num_edges = 45
                # DEBUG: automaton num_states = 8, num_edges = 27
            },
            {
                "agent_0": '!green_0 U (yellow_0 & F (!(yellow_0 | magenta_0) U (blue_0 & F magenta_0)))',
                "agent_1": '!magenta_1 U ((green_1 | blue_1) & F (!yellow_1 U (magenta_1 & F green_1)))',
                "agent_2": '!(green_2 | yellow_2) U (magenta_2 & F (!magenta_2 U (blue_2 & F yellow_2)))',
                "req": "(yellow_0 Req magenta_2) & (yellow_0 Req magenta_1) & (magenta_2 Req blue_0) & (magenta_1 Req blue_2)",
                "cnt": "Sync[{1, 2}, green, 1] & Async[{0, 1, 2}, blue, 2] & Sync[{0, 1}, magenta, 1]",
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 5, num_edges = 12
                # DEBUG: automaton num_states = 10, num_edges = 45
                # DEBUG: automaton num_states = 32, num_edges = 243 
            },
        ],
        "ma3_ind_reach_only": 
        [
            {
                "agent_0": "(F magenta_0)",
                "agent_1": "(F blue_1)",
                "agent_2": "(F green_2)",
            },
            {
                "agent_0": "(F green_0)",
                "agent_1": "(F magenta_1)",
                "agent_2": "(F magenta_2)",
            },
            {
                "agent_0": "(F blue_0)",
                "agent_1": "(F yellow_1)",
                "agent_2": "(F blue_2)",
            },
        ],
        "ma3_ind_reach_avoid": 
        [
            {
                "agent_0": "!magenta_0 U blue_0",
                "agent_1": "!blue_1 U green_1",
                "agent_2": "!green_2 U magenta_2",
            },
            {
                "agent_0": "!green_0 U yellow_0",
                "agent_1": "!magenta_1 U blue_1",
                "agent_2": "!magenta_2 U green_2",
            },
            {
                "agent_0": "!blue_0 U magenta_0",
                "agent_1": "!yellow_1 U green_1",
                "agent_2": "!blue_2 U yellow_2",
            },
        ],
        "ma5_ind_req_cnt":
        [
            {
                # 5 agents
                "agent_0": '(!magenta_0 U (blue_0 | green_0)) & (F magenta_0)', # phi_1
                "agent_1": '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))', # phi_2
                "agent_2": '!(magenta_2 | yellow_2) U (blue_2 & F green_2)', # phi_3
                "agent_3": '!magenta_3 U ((blue_3 | yellow_3) & F green_3)', # phi_4
                "agent_4": '!(blue_4 | yellow_4) U (green_4 & F magenta_4)', # phi_5
                "req": "(blue_0 Req green_1) & (blue_1 Req green_2) & (green_2 Req yellow_3) & (green_3 Req green_4)",
                "cnt": "Async[{0, 1, 2, 3, 4}, yellow, 3] & Sync[{0, 1, 2, 3, 4}, blue, 2]",
            },
            {
                # 5 agents
                "agent_0": '!(blue_0 | yellow_0) U (green_0 & F magenta_0)', # phi_5
                "agent_1": '(F magenta_1) & (!yellow_1 U (blue_1 | green_1))', # phi_6
                "agent_2": '!green_2 U (yellow_2 & F (blue_2 | magenta_2))', # phi_7
                "agent_3": '!magenta_3 U ((green_3 | blue_3) & F yellow_3)', # phi_8
                "agent_4": '!(green_4 | yellow_4) U (magenta_4 & F blue_4)', # phi_9
                "req": "(green_0 Req magenta_1) & (blue_1 Req blue_2) & (yellow_2 Req green_3) & (blue_3 Req yellow_4)",
                "cnt": "Sync[{0, 1, 2, 3, 4}, yellow, 2] & Async[{0, 1, 2, 3, 4}, green, 4]",
            },
            {
                # 5 agents
                "agent_0": '(!magenta_0 U (blue_0 | green_0)) & (F magenta_0)', # phi_1
                "agent_1": '!(magenta_1 | yellow_1) U (blue_1 & F green_1)', # phi_3
                "agent_2": '!(blue_2 | yellow_2) U (green_2 & F magenta_2)', # phi_5
                "agent_3": '!green_3 U (yellow_3 & F (blue_3 | magenta_3))', # phi_7
                "agent_4": '!(green_4 | yellow_4) U (magenta_4 & F blue_4)', # phi_9
                "req": "(blue_0 Req blue_1) & (green_1 Req green_2) & (magenta_2 Req yellow_3) & (yellow_3 Req magenta_4)",
                "cnt": "Sync[{0, 1, 2, 3, 4}, magenta, 2] & Sync[{0, 1, 2, 3, 4}, green, 2]",
            },
        ],
        "ma10_ind_req_cnt":
        [
            {
                'agent_0': '!(magenta_0 | yellow_0) U (blue_0 & F green_0)', 
                'agent_1': '!magenta_1 U ((green_1 | blue_1) & F yellow_1)', 
                'agent_2': '(!magenta_2 U (blue_2 | green_2)) & (F magenta_2)', 
                'agent_3': '(F magenta_3) & (!yellow_3 U (blue_3 | green_3))', 
                'agent_4': '!magenta_4 U ((blue_4 | yellow_4) & F green_4)', 
                'agent_5': '!(green_5 | yellow_5) U (magenta_5 & F blue_5)', 
                'agent_6': '(F blue_6) & (!blue_6 U (green_6 & F yellow_6))', 
                'agent_7': '!(blue_7 | yellow_7) U (green_7 & F magenta_7)', 
                'agent_8': '!green_8 U (yellow_8 & F (blue_8 | magenta_8))', 
                'agent_9': '!(magenta_9 | yellow_9) U (blue_9 & F green_9)',
                'req': "(blue_0 Req green_1) & (blue_1 Req green_2) & "
                        "(blue_2 Req magenta_3) & (blue_3 Req blue_4) & "
                        "(green_4 Req magenta_5) & (blue_5 Req green_6) & "
                        "(blue_6 Req green_7) & (magenta_7 Req blue_8) & (yellow_8 Req blue_9)",
                # 'cnt': "Sync[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, blue, 4]",
                'cnt': "Async[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, blue, 9]",
            },
            {
                'agent_0': '!green_0 U (yellow_0 & F (blue_0 | magenta_0))', 
                'agent_1': '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))', 
                'agent_2': '!(green_2 | yellow_2) U (magenta_2 & F blue_2)', 
                'agent_3': '!magenta_3 U ((blue_3 | yellow_3) & F green_3)', 
                'agent_4': '(!magenta_4 U (blue_4 | green_4)) & (F magenta_4)', 
                'agent_5': '(F magenta_5) & (!yellow_5 U (blue_5 | green_5))', 
                'agent_6': '!magenta_6 U ((green_6 | blue_6) & F yellow_6)', 
                'agent_7': '!(magenta_7 | yellow_7) U (blue_7 & F green_7)', 
                'agent_8': '!(blue_8 | yellow_8) U (green_8 & F magenta_8)', 
                'agent_9': '!magenta_9 U ((green_9 | blue_9) & F yellow_9)',
                "req": "(yellow_0 Req green_1) & (yellow_1 Req magenta_2) & "
                        "(blue_2 Req blue_3) & (green_3 Req blue_4) & "
                        "(blue_4 Req blue_5) & (green_5 Req green_6) & "
                        "(yellow_6 Req blue_7) & (green_7 Req green_8) & "
                        "(magenta_8 Req green_9)",
                # "cnt": "Sync[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, magenta, 3]",
                "cnt": "Async[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, magenta, 7]",
            },
            {
                'agent_0': '!(blue_0 | yellow_0) U (green_0 & F magenta_0)', 
                'agent_1': '(!magenta_1 U (blue_1 | green_1)) & (F magenta_1)', 
                'agent_2': '!green_2 U (yellow_2 & F (blue_2 | magenta_2))', 
                'agent_3': '!(magenta_3 | yellow_3) U (blue_3 & F green_3)', 
                'agent_4': '!(green_4 | yellow_4) U (magenta_4 & F blue_4)', 
                'agent_5': '(F blue_5) & (!blue_5 U (green_5 & F yellow_5))', 
                'agent_6': '(F magenta_6) & (!yellow_6 U (blue_6 | green_6))', 
                'agent_7': '!magenta_7 U ((blue_7 | yellow_7) & F green_7)', 
                'agent_8': '!magenta_8 U ((green_8 | blue_8) & F yellow_8)', 
                'agent_9': '(!magenta_9 U (blue_9 | green_9)) & (F magenta_9)',
                "req": "(magenta_0 Req blue_1) & (magenta_1 Req yellow_2) & "
                        "(yellow_2 Req blue_3) & (blue_3 Req magenta_4) & "
                        "(blue_4 Req yellow_5) & (yellow_5 Req magenta_6) & "
                        "(magenta_6 Req blue_7) & (green_7 Req yellow_8) & "
                        "(yellow_8 Req magenta_9)",
                # 'cnt': "Sync[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, green, 5]"
                'cnt': "Async[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, green, 9]"
            },
        ],
        "ma10_ind":
        [
            {
                'agent_0': '!(magenta_0 | yellow_0) U (blue_0 & F green_0)', 
                'agent_1': '!magenta_1 U ((green_1 | blue_1) & F yellow_1)', 
                'agent_2': '(!magenta_2 U (blue_2 | green_2)) & (F magenta_2)', 
                'agent_3': '(F magenta_3) & (!yellow_3 U (blue_3 | green_3))', 
                'agent_4': '!magenta_4 U ((blue_4 | yellow_4) & F green_4)', 
                'agent_5': '!(green_5 | yellow_5) U (magenta_5 & F blue_5)', 
                'agent_6': '(F blue_6) & (!blue_6 U (green_6 & F yellow_6))', 
                'agent_7': '!(blue_7 | yellow_7) U (green_7 & F magenta_7)', 
                'agent_8': '!green_8 U (yellow_8 & F (blue_8 | magenta_8))', 
                'agent_9': '!(magenta_9 | yellow_9) U (blue_9 & F green_9)',
            },
            {
                'agent_0': '!green_0 U (yellow_0 & F (blue_0 | magenta_0))', 
                'agent_1': '(F blue_1) & (!blue_1 U (green_1 & F yellow_1))', 
                'agent_2': '!(green_2 | yellow_2) U (magenta_2 & F blue_2)', 
                'agent_3': '!magenta_3 U ((blue_3 | yellow_3) & F green_3)', 
                'agent_4': '(!magenta_4 U (blue_4 | green_4)) & (F magenta_4)', 
                'agent_5': '(F magenta_5) & (!yellow_5 U (blue_5 | green_5))', 
                'agent_6': '!magenta_6 U ((green_6 | blue_6) & F yellow_6)', 
                'agent_7': '!(magenta_7 | yellow_7) U (blue_7 & F green_7)', 
                'agent_8': '!(blue_8 | yellow_8) U (green_8 & F magenta_8)', 
                'agent_9': '!magenta_9 U ((green_9 | blue_9) & F yellow_9)',
            },
            {
                'agent_0': '!(blue_0 | yellow_0) U (green_0 & F magenta_0)', 
                'agent_1': '(!magenta_1 U (blue_1 | green_1)) & (F magenta_1)', 
                'agent_2': '!green_2 U (yellow_2 & F (blue_2 | magenta_2))', 
                'agent_3': '!(magenta_3 | yellow_3) U (blue_3 & F green_3)', 
                'agent_4': '!(green_4 | yellow_4) U (magenta_4 & F blue_4)', 
                'agent_5': '(F blue_5) & (!blue_5 U (green_5 & F yellow_5))', 
                'agent_6': '(F magenta_6) & (!yellow_6 U (blue_6 | green_6))', 
                'agent_7': '!magenta_7 U ((blue_7 | yellow_7) & F green_7)', 
                'agent_8': '!magenta_8 U ((green_8 | blue_8) & F yellow_8)', 
                'agent_9': '(!magenta_9 U (blue_9 | green_9)) & (F magenta_9)',
            },
        ],
    }
}


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic=True, key=None):

    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    sampler = FixedSampler.partial(formula)
    max_steps = None
    print(f"DEBUG: max_steps = {max_steps}, deterministic = {deterministic}, seed = {seed}")
    
    env = make_env_safety(env_name, sampler, max_steps, render_mode='human' if render else None)
    config = model_configs[env_name]
    
    model_store = ModelStore(env_name, exp, seed, None)
    training_status = model_store.load_training_status(map_location='cpu')
    model = build_model_safety(env, training_status, config)
    
    props = env.get_propositions()
    search = SpotBFS(env, model, props, num_loops=1)
    agent = MultiAgent(env, model, search=search, propositions=props, seed=seed, verbose=render)

    num_successes = 0
    num_violations = 0
    num_collisions = 0
    num_unreachable = 0
    num_accepting_visits = []
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset()
        if render:
            print(obs['goal'])
        agent.reset()
        done = {a: False for a in env.possible_agents}
        num_steps = 0
        while not any(done.values()):
            action = agent.get_action(obs, info, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                action = action.flatten()
                if action.shape == (1,):
                    action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if any(done.values()):
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                        if 'violation_collision' in info:
                            num_collisions += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'CV': num_collisions / num_violations if num_violations > 0 else 0,
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    if 'violation' in info:
                        num_violations += 1
                        if 'violation_collision' in info:
                            num_collisions += 1
                    else:
                        num_accepting_visits.append(info['num_accepting_visits'])
                    if not render:
                        pbar.set_postfix({
                            'A': np.mean(num_accepting_visits),
                            'V': num_violations / (i + 1),
                            'CV': num_collisions / num_violations if num_violations > 0 else 0,
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        collision_rate = num_collisions / num_violations if num_violations > 0 else 0
        unreachable_rate = num_unreachable / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        print(f'{seed}: success_rate={success_rate:.3f},violation_rate={violation_rate:.3f},collision_rate={collision_rate:.3f},unreachable_rate={unreachable_rate:.3f},adr={adr:.3f},average_steps={average_steps:.3f}')
        return num_successes, num_violations, num_collisions, average_steps
    else:
        average_visits = np.mean(num_accepting_visits)
        violation_rate = num_violations / num_episodes
        collision_rate = num_collisions / num_violations if num_violations > 0 else 0
        unreachable_rate = num_unreachable / num_episodes
        print(f'{seed}: average_visits={average_visits:.3f},violation_rate={violation_rate:.3f},collision_rate={collision_rate:.3f},unreachable_rate={unreachable_rate:.3f}')
        return average_visits, num_violations, num_collisions



def main(env, exp, seeds, key, num_episodes=100):
    base_env = env.split('.')[0]  # e.g. SafetyPointMultiGoal0-v0.ma3 -> SafetyPointMultiGoal0-v0
    tasks = env_to_tasks[base_env][key]
    gamma = {
        'PointLtl2-v0': 0.998,
        'PointLtlSafety2-v0': 0.998,
        'PointLtlSafety3-v0': 0.998,
        'PointLtlSafety4-v0': 0.998,
        'PointLtlSafety5-v0': 0.998,
        'LetterEnv-v0': 0.94,
        'LetterSafetyEnv-v0': 0.94,
        "SafetyPointMultiGoal0-v0": 0.998,
    }[base_env]
    
    num_procs = len(seeds) * len(tasks)
    file_path = f'results_finite/{env}_{exp}_{key}.csv'
    print(f"save to {file_path}")
    with mp.Pool(num_procs) as pool:
        args = []
        for task in tasks:
            print(f'Running task: {task}')
            for seed in seeds:
                args.append([env, gamma, exp, seed, num_episodes, task, True, False, True, key])
        for result in tqdm(pool.imap_unordered(eval_task, args), total=len(seeds)*len(tasks)):
            results = []
            (sn, vn, cn, mean_steps), seed, task = result
            results.append([exp, task, seed, sn, vn, cn, mean_steps])
            df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'success_rate', 'violation_rate', 'collision_rate', 'mean_steps'])
            os.makedirs('results_finite', exist_ok=True)

            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3], simulate_args[5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SafetyPointMultiGoal0-v0.ma3')
    parser.add_argument('--exp', type=str, default='GenZ-LTL')
    parser.add_argument('--seed', type=int, nargs='+', default=[1], help="List of seeds")
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()
    main(args.env, args.exp, args.seed, args.key, args.num_episodes)
