# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from copy import copy
from data import QValue, DataTable, Result
from itertools import chain
from search import merge_eval_info
from typing import Optional, List, Iterable, Callable


class Agent(ABC):
    @abstractmethod
    def done(self) -> bool:
        """
        :return: If the agent has done searching.
        """
        pass

    @abstractmethod
    def table(self) -> DataTable:
        """
        :return: On which table the agent is working.
        """
        pass

    @abstractmethod
    def ranked_complete_states(self) -> List[Result]:
        """
        :return: Current complete results.
        """
        pass

    @abstractmethod
    def step(self) -> List[QValue]:
        """
        Take a step forward: (Initialize and) Search the states for estimation.
        :return: Chosen (state, actions) pairs for model prediction/scoring.
        """
        pass

    @staticmethod
    def valid_results(chosen: List[QValue], predicted_values: Iterable) -> Iterable[Result]:
        for state_actions, action_values in zip(chosen, predicted_values):
            state = state_actions.state
            actions = state_actions.actions
            valid_mask = state_actions.valid_mask

            for action, valid, score in zip(actions, valid_mask, action_values[:len(actions)]):
                if not valid:
                    continue
                result = Result(score, copy(state).append(action))
                yield result

    @abstractmethod
    def update(self, chosen: List[QValue], predicted_values: Iterable) -> Optional[dict]:
        """
        Consume the estimation results.
        :param chosen: The chosen pairs returned by step().
        :param predicted_values: Predicted action values for the chosen pairs.
        :return: Recorder summary dict when the agent is done, else None.
        """
        pass


class ParallelAgents:
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.agents = []
        self.chosen_lists = None
        self.info_list = []
        self.error_cnt = 0

    def add(self, agent: Agent):
        if self.chosen_lists is not None:
            raise ValueError("Previous step not updated!")
        self.agents.append(agent)

    def extend(self, agents: Iterable[Agent]):
        if self.chosen_lists is not None:
            raise ValueError("Previous step not updated!")
        self.agents.extend(agents)

    def finished(self) -> int:
        return len(self.info_list) + self.error_cnt

    def remaining(self) -> int:
        return len(self.agents)

    @staticmethod
    def _step_(agent: Agent) -> List[QValue]:
        try:
            return agent.step()
        except ValueError:  # Already finished or No user created PivotTable!
            return []

    def step(self, feed_batch_fns: List[Callable]) -> List[Future]:
        if self.chosen_lists is not None:
            raise ValueError("Previous step not updated!")
        self.chosen_lists = list(self.executor.map(self._step_, self.agents))
        # print([[chosen.state for chosen in chosen_list] for chosen_list in self.chosen_lists])
        n_fns = len(feed_batch_fns)
        futures = []
        for i, feed_batch_fn in enumerate(feed_batch_fns):
            samples = list(chain.from_iterable(self.chosen_lists[i::n_fns]))  # WARNING: samples could be EMPTY!
            futures.append(self.executor.submit(feed_batch_fn, samples))
        return futures

    def _update_(self, i: int, n_fns: int, results_get_fn: Callable):
        results = results_get_fn()  # Get numpy array
        start = 0
        info_list = []
        for j in range(i, len(self.chosen_lists), n_fns):
            end = start + len(self.chosen_lists[j])
            try:
                info = self.agents[j].update(self.chosen_lists[j], results[start:end])
                if info is not None:
                    info_list.append(info)
            except ValueError:  # Already finished or No user created PivotTable!
                self.error_cnt += 1
            start = end
        return info_list

    def update(self, results_get_fns: List[Callable]):
        n_fns = len(results_get_fns)
        futures = []
        for i, results_get_fn in enumerate(results_get_fns):
            futures.append(self.executor.submit(self._update_, i, n_fns, results_get_fn))
        finished_info = []
        for future in futures:
            finished_info.extend([info for info in future.result() if info])
        # old_len = len(self.agents)
        self.agents = [agent for agent in self.agents if not agent.done()]  # Remove finished agents
        # print("Removed {} agents.".format(old_len - len(self.agents)))
        self.chosen_lists = None
        self.info_list.extend(finished_info)
        return finished_info

    def shutdown(self):
        self.executor.shutdown()

    def summary(self, divide_total: bool = True):
        return merge_eval_info(self.info_list, divide_total)
