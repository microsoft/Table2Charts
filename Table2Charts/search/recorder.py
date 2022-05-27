# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from collections import defaultdict
from data import State, DataTable, Result
from sortedcontainers import SortedList
from time import process_time, perf_counter
from typing import Optional, Set, List, Tuple, Iterable


class Evaluation:
    def __init__(self, rank_states: List[State], targets: Set[State], top: Tuple[int],
                 test_field_selections: bool, test_design_choices: bool):
        if test_design_choices and test_field_selections:
            raise ValueError(
                "Parameters test_design_choices and test_field_selections could not be true at the same time.")

        self.reached = len(rank_states)
        # unordered_targets = set(UnorderedState(state) for state in targets)
        self.target_cnt = len(targets)
        # self.target_unordered_cnt = len(unordered_targets)
        self.top = sorted(top)
        self.total = 0

        k = 0
        hit_ordered = 0
        # hit_unordered = 0
        self.pos_ordered = -1
        # self.pos_unordered = -1
        self.hit_cnt = []

        if test_design_choices:
            target_fields = list(set(tuple(x.unique_field_tokens()) for x in targets))
            recall_record = [0] * len(self.top)
            for target_field in target_fields:
                hit_task1 = False
                hit_task2 = False
                i = 0  # order of correct field selection
                k = 0
                for state in rank_states:
                    if len(set(state.unique_field_tokens()).difference(set(target_field))) == 0:
                        hit_task1 = True
                        hit_task2 = state in targets
                        # hit_task2 = any([state.tokens[1:] == target.tokens[1:] for target in list(targets)])  # field mapping
                        # hit_task2 = any([state.tokens[0] == target.tokens[0] for target in list(targets)])  # chart type
                        if k < len(self.top) and i == self.top[k] - 1:
                            recall_record[k] += 1 if hit_task2 else 0
                            k += 1
                        i += 1
                        if hit_task2:
                            hit_ordered += 1
                            if self.pos_ordered == -1:
                                self.pos_ordered = i + 1
                            else:
                                self.pos_ordered = min(i + 1, self.pos_ordered)
                            break
                while k < len(self.top):
                    recall_record[k] += 1 if hit_task2 else 0
                    k += 1
            self.total = len(target_fields)
            for k in range(len(self.top)):
                self.hit_cnt.append((f"@{self.top[k]:02d}", recall_record[k]))
            self.hit_cnt.append(("all", hit_ordered))

        else:
            self.total = 0 if len(targets) is 0 else 1
            for i, state in enumerate(rank_states):
                if test_field_selections:
                    for target in targets:
                        if len(set(state.unique_field_tokens()).difference(set(target.unique_field_tokens()))) == 0:
                            hit_ordered += 1
                            if self.pos_ordered == -1:
                                self.pos_ordered = i + 1
                            break
                else:
                    if state in targets:
                        hit_ordered += 1
                        # hit_unordered += 1
                        if self.pos_ordered == -1:
                            self.pos_ordered = i + 1
                            # self.pos_unordered = i + 1
                    # elif UnorderedState(state) in unordered_targets:
                    #     hit_unordered += 1
                    #     if self.pos_unordered == -1:
                    #         self.pos_unordered = i + 1

                if k < len(self.top) and i == self.top[k] - 1:
                    self.hit_cnt.append((f"@{self.top[k]:02d}", 1 if hit_ordered > 0 else 0))
                    # self.hit_cnt.append((f"@{self.top[k]:02d}", (hit_ordered, hit_unordered)))
                    k += 1
            while k < len(self.top):
                self.hit_cnt.append((f"@{self.top[k]:02d}", 1 if hit_ordered > 0 else 0))
                # self.hit_cnt.append((f"@{self.top[k]:02d}", (hit_ordered, hit_unordered)))
                k += 1
            # TODO: also guarantee all stages are presented
            self.hit_cnt.append(("all", 1 if hit_ordered > 0 else 0))
        # self.hit_cnt.append(("all", (hit_ordered, hit_unordered)))

        # TODO: implement edit distance
        # TODO: implement values coverage rate
        # TODO: implement value prediction metrics
        # TODO: implement dimension combination recall

    def to_json(self):
        return {
            "recall": dict(self.hit_cnt),
            "first_rank": self.pos_ordered,  # (self.pos_ordered, self.pos_unordered),
            "reached": self.reached,
            "targets": self.target_cnt,  # (self.target_cnt, self.target_unordered_cnt),
            # TODO: add average depth
            # TODO: add average stage
            "top": self.top,
            "total": self.total
        }


class Recorder:
    """Record the searched states for the process of heuristic searching on a DataTable."""
    COMPLETE = "complete"

    def __init__(self, table: DataTable, targets: Optional[Set[State]] = None,
                 log_path: Optional[str] = None, top: Tuple[int] = (1, 3, 5, 10, 20),
                 test_field_selections: bool = False, test_design_choices: bool = False):
        if targets and len(targets) == 0:
            raise ValueError("No user created targets in {}!".format(table.tUID))
        self.tUID = table.tUID
        self.width = table.n_cols
        self.targets = targets
        self.top = top
        self.test_field_selections = test_field_selections
        self.test_design_choices = test_design_choices

        # Record states
        self.expanded_states = 0
        self.cut_states = 0
        self.dropped_states = 0
        self.reached_states = 0
        self.stage_results = defaultdict(lambda: SortedList([], key=lambda x: -x.score))
        self.depth_results = defaultdict(lambda: SortedList([], key=lambda x: -x.score))

        # Record time
        self.start_process_time = process_time()
        self.start_perf_time = perf_counter()
        self.end_process_time = None
        self.end_perf_time = None

        # Logging
        self.info = dict()
        self.log_path = log_path

        self.finished = False

    def record_reached(self, states: Iterable[Result]):
        for result in states:
            self.reached_states += 1
            score, state = result
            if state.is_stage_checkpoint():
                self.stage_results[self._parts_str_(state.stage())].add(result)
            if state.is_complete():
                self.stage_results[self.COMPLETE].add(result)
            self.depth_results[len(state)].add(result)

    def count_expanded(self, cnt: int):
        self.expanded_states += cnt

    def count_cut(self, cnt: int):
        self.cut_states += cnt

    def count_dropped(self, cnt: int):
        self.dropped_states += cnt

    def start(self):
        self.start_process_time = process_time()
        self.start_perf_time = perf_counter()
        self.finished = False

    def passed_time(self):
        if self.finished:
            return self.end_process_time - self.start_process_time, self.end_perf_time - self.start_perf_time
        else:
            return process_time() - self.start_process_time, perf_counter() - self.start_perf_time

    def end(self, is_single_inference: bool = False):
        self.finished = True
        self.end_process_time = process_time()
        self.end_perf_time = perf_counter()

        complete_results = self.completed_results()
        process_t, perf_t = self.passed_time()
        info = {
            "tUID": self.tUID,
            "process_time": process_t,
            "perf_time": perf_t,
            "reached_states": self.reached_states,
            "expanded_states": self.expanded_states,
            "cut_states": self.cut_states,
            "dropped_states": self.dropped_states,
            "complete_states": len(complete_results),
        }

        if self.targets is not None:
            stage_targets = defaultdict(set)
            depth_targets = defaultdict(set)
            for target in self.targets:
                if target.is_complete():
                    stage_targets[self.COMPLETE].add(target)
                if target.is_stage_checkpoint():
                    stage_targets[target.stage()].add(target)
                depth_targets[len(target)].add(target)
            info["evaluation"] = {
                "stages": {
                    stage: Evaluation([r[1] for r in results], stage_targets[stage], self.top,
                                      self.test_field_selections, self.test_design_choices).to_json()
                    for stage, results in self.stage_results.items()
                },
                "depths": {
                    depth: Evaluation([r[1] for r in results], depth_targets[depth], self.top,
                                      self.test_field_selections, self.test_design_choices).to_json()
                    for depth, results in self.depth_results.items()
                },
                "width": {
                    self.width: Evaluation([r[1] for r in complete_results], stage_targets[self.COMPLETE], self.top,
                                           self.test_field_selections, self.test_design_choices).to_json()
                }
            }
            info["user_created"] = [state.to_json() for state in stage_targets[self.COMPLETE]]

        if self.log_path or is_single_inference:
            saved_info = dict()
            rank = []
            for score, state in complete_results:
                dic = state.to_json()
                dic["score"] = float(score)
                rank.append(dic)
            info["ranked_recommend"] = rank
            saved_info["ranked_recommend"] = rank

            if is_single_inference:
                return saved_info
            else:
                saved_info["tUID"] = info["tUID"]
                saved_info["user_created"] = info["user_created"] if self.targets is not None else []
                saved_info["num_fields"] = self.width
                os.makedirs(self.log_path, exist_ok=True)
                with open(os.path.join(self.log_path, self.tUID + ".json"), "w") as f:
                    json.dump(saved_info, f)  # All in ASCII

        return info

    @staticmethod
    def _parts_str_(field_parts: int):
        return f"{field_parts:02d}p"

    def staged_results(self, field_parts: int):
        return self.stage_results[self._parts_str_(field_parts)]

    def completed_results(self):
        return self.stage_results[self.COMPLETE]

    def all_results(self):
        r = SortedList([], key=lambda x: -x[0])
        for depth, results in self.depth_results.items():
            r.update(results)
        return r


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return "{:.2f}*{}".format(self.avg, self.count)


def _reduce_entries_(info_list: List[dict], keys: List[str], out: dict,
                     divide_total: bool = True, zero_one_only: bool = False, filter_negative: bool = False,
                     total: int = 0, is_total: bool = False):
    pair = lambda summation, cnt: summation / cnt if divide_total else (summation, cnt)
    if len(info_list) == 0:
        return len(info_list)
    for key in keys:
        samples = [info[key] for info in info_list if key in info]

        # When the former iterations don't meet "total", use len.
        if not is_total:
            if "total" in info_list[0]:
                total = sum([info["total"] for info in info_list if key in info])
                is_total = True
            else:
                total = len(samples)

        if total == 0:
            continue

        if isinstance(samples[0], dict):
            out[key] = dict()
            merged_keys = set()
            for sample in samples:
                merged_keys.update(sample.keys())
            _reduce_entries_(samples, list(merged_keys), out[key],
                             divide_total=divide_total, zero_one_only=zero_one_only, filter_negative=filter_negative,
                             total=total, is_total=is_total)
        # elif isinstance(samples[0], tuple) or isinstance(samples[0], list):
        #     length = len(samples[0])
        #     if zero_one_only:
        #         out[key] = tuple(pair(sum(1 if sample[i] > 0 else 0 for sample in samples), total)
        #                          for i in range(length))
        #     elif filter_negative:
        #         t = []
        #         for i in range(length):
        #             meter = AverageMeter()
        #             for sample in samples:
        #                 if sample[i] >= 0:  # E.g. -1 for first_rank
        #                     meter.update(sample[i])
        #             t.append(str(meter))
        #         out[key] = tuple(t)
        #     else:
        #         out[key] = tuple(pair(sum(sample[i] for sample in samples), total)
        #                          for i in range(length))
        else:
            if zero_one_only:
                out[key] = pair(sum(samples), total)
            elif filter_negative:  # TODO: can be used in task2?
                meter = AverageMeter()
                for sample in samples:
                    if sample >= 0:  # E.g. -1 for first_rank
                        meter.update(sample)
                out[key] = str(meter)
            else:
                out[key] = pair(sum(samples), total)
    return len(info_list)


def _count_and_merge_eval_(t_evaluations: Iterable[dict], divide_total: bool = True):
    cat_eval_list = defaultdict(list)
    for t_evaluation in t_evaluations:
        for cat, evaluation in t_evaluation.items():
            cat_eval_list[cat].append(evaluation)

    merged = dict()
    for cat, evaluations in cat_eval_list.items():
        merged[cat] = dict()
        _reduce_entries_(evaluations, ["recall"], merged[cat], zero_one_only=True, divide_total=divide_total)
        _reduce_entries_(evaluations, ["first_rank"], merged[cat], filter_negative=True, divide_total=divide_total)
        t_cnt = _reduce_entries_(evaluations, ["reached", "targets"], merged[cat], divide_total=divide_total)
        merged[cat]["top"] = evaluations[0]["top"]
        merged[cat]["t_cnt"] = t_cnt

    return merged


def merge_eval_info(info_list: List[dict], divide_total: bool = True):
    result = dict()
    t_cnt = _reduce_entries_(info_list, [
        "process_time", "perf_time", "reached_states", "expanded_states",
        "cut_states", "dropped_states", "complete_states"], result, divide_total=divide_total)
    result["t_cnt"] = t_cnt

    if len(info_list) != 0 and "evaluation" in info_list[0]:
        result["evaluation"] = {
            "stages": _count_and_merge_eval_([i["evaluation"]["stages"] for i in info_list], divide_total=divide_total),
            "depths": _count_and_merge_eval_([i["evaluation"]["depths"] for i in info_list], divide_total=divide_total),
            "width": _count_and_merge_eval_([i["evaluation"]["width"] for i in info_list], divide_total=False)
        }

    return result
