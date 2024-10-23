import logging
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from simglucose.controller.rlpid_ctrller import RLPIDController
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path

    def simulate(self):
        obs, reward, done, info = self.env.reset()
        if isinstance(self.controller, RLPIDController):
            self.controller.reset(obs, **info)
        tic = time.time()
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            action, kp, ki, kd = self.controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)
            if isinstance(self.controller, RLPIDController):
                # Update the history lists and create the new state representation
                self.controller.bg_history.pop(0)
                self.controller.bg_history.append(obs.CGM)
                self.controller.meal_history.pop(0)
                self.controller.meal_history.append(info.get('meal'))

                # Create the action tensor using kp, ki, kd
                action = torch.tensor([kp, ki, kd], dtype=torch.float32)
                self.controller.step_and_record(obs, action, reward, done, **info)

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.env.viewer.fig.savefig(os.path.join(self.path, f"{str(self.env.patient.name)}.png"))
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

    def results(self):
        return self.env.show_history()

    def save_results(self):
        df = self.results()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        filename = os.path.join(self.path, str(self.env.patient.name) + '.csv')
        df.to_csv(filename)

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    sim_object.save_results()
    print('Simulation Completed!')
    return sim_object.results()


def batch_sim(sim_instances, parallel=False):
    tic = time.time()
    if parallel and pathos:
        with Pool() as p:
            results = p.map(sim, sim_instances)
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s) for s in sim_instances]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results
