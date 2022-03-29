import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from dataLoad import *

from frankwolfe import flow

class TrafficEnvironment(py_environment.PyEnvironment):
    def __init__(self,demand,edges,perturbed,fake_flow,real_flow,pert_indices,limit=100):
        self.demand=demand
        self.edges=edges
        self.initial_perturbed=perturbed
        self.perturbed=self.initial_perturbed.copy()
        self.fake_flow=fake_flow
        self.real_flow=real_flow
        self.pert_indices=pert_indices
        edges_count=len(fake_flow["flow"])
        self._action_spec = array_spec.BoundedArraySpec(shape=(edges_count,), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(edges_count,), dtype=np.float32, minimum=0, name='observation')
        self._initial_state= [f for f in fake_flow["flow"]]
        self._state = self._initial_state.copy()
        self._episode_ended = False
        self.step_count=0
        self.limit=limit
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = self._initial_state.copy()
        self.fake_flow["flow"]=self._initial_state.copy()
        self.perturbed=self.initial_perturbed.copy()
        self._episode_ended = False
        self.step_count=0
        return ts.restart(np.array(self._state, dtype=np.float32))
    
    def _step(self, action):
    
        if self._episode_ended:
            return self.reset()
        self.step_count+=1
        # Make sure episodes don't go on forever.
        if self.step_count>self.limit:
            self._episode_ended=True
        m=0
        try:
            self._state=flow(self.demand,self.perturbed,25)["flow"]
        except TypeError:
            for k,v in self.demand.items():
                for val in v:
                    m=max(m,val)
                    if type(val) != type(1):
                        print(k,val)
            for k,v in self.perturbed.items():
                for val in v:
                    m=max(m,val)
                    if type(val) != type(1):
                        print(k,val)
            print("yes type error max value was ",m)
            exit()
        diff=[]
        for a,b in zip(self.real_flow["flow"],self._state):
            diff.append(np.abs(a-b))
        reward=-np.linalg.norm(diff) #regularization term we want norm(Y) and norm(predicted edges) to be minimized 
        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            for x in range(len(action)):
                self.perturbed["capacity"][x]=int(750*action[x])+250
            return ts.transition(np.array(self._state, dtype=np.float32), reward=0.0, discount=1.0)