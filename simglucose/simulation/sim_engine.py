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
            self.controller.reset(obs , **info)
        tic = time.time()
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            action , kp , ki ,kd = self.controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)
            if isinstance(self.controller, RLPIDController):
                action = torch.tensor([kp, ki, kd], dtype=torch.float32)
                self.controller.step_and_record(obs , action , reward , done , **info)

            # self.controller
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

    # def generateFinalPlot(self):
    #     data = self.results()
    #     start_time = self.env.scenario.start_time
    #     patient_name  = self.env.patient.name


    #     fig, axes = plt.subplots(4, figsize=(10, 8))

    #     # Plot Blood Glucose (BG) and CGM
    #     axes[0].set_ylabel('BG (mg/dL)')
    #     axes[0].plot(data.index, data['BG'], label='BG')
    #     axes[0].plot(data.index, data['CGM'], label='CGM')
    #     axes[0].legend()
    #     axes[0].set_ylim([70, 180])
    #     axes[0].axhspan(70, 180, alpha=0.3, color='limegreen', lw=0)
    #     axes[0].axhspan(50, 70, alpha=0.3, color='red', lw=0)
    #     axes[0].axhspan(0, 50, alpha=0.3, color='darkred', lw=0)
    #     axes[0].axhspan(180, 250, alpha=0.3, color='red', lw=0)
    #     axes[0].axhspan(250, 1000, alpha=0.3, color='darkred', lw=0)
    #     axes[0].set_xlim([start_time, data.index[-1]])

    #     # Plot Carbohydrate (CHO)
    #     axes[1].set_ylabel('CHO (g/min)')
    #     axes[1].plot(data.index, data['CHO'], label='CHO')
    #     axes[1].legend()
    #     axes[1].set_ylim([-5, 30])
    #     axes[1].set_xlim([start_time, data.index[-1]])

    #     # Plot Insulin
    #     axes[2].set_ylabel('Insulin (U/min)')
    #     axes[2].plot(data.index, data['insulin'], label='Insulin')
    #     axes[2].legend()
    #     axes[2].set_ylim([-0.5, 1])
    #     axes[2].set_xlim([start_time, data.index[-1]])

    #     # Plot Risk Index, LBGI, and HBGI
    #     axes[3].set_ylabel('Risk Index')
    #     axes[3].plot(data.index, data['LBGI'], label='Hypo Risk')
    #     axes[3].plot(data.index, data['HBGI'], label='Hyper Risk')
    #     axes[3].plot(data.index, data['Risk'], label='Risk Index')
    #     axes[3].legend()
    #     axes[3].set_ylim([0, 5])
    #     axes[3].set_xlim([start_time, data.index[-1]])

    #     # X-axis formatting
    #     axes[3].xaxis.set_minor_locator(mdates.AutoDateLocator())
    #     axes[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    #     axes[3].xaxis.set_major_locator(mdates.DayLocator())
    #     axes[3].xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    #     axes[0].set_title(patient_name)

    #     fig.tight_layout()

    #     fig.savefig(os.path.join(self.path, f"{str(self.env.patient.name)}.png"))

        # return fig, axes


    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    sim_object.save_results()
    # sim_object.generateFinalPlot()
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
