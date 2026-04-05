#!/usr/bin/env python3
"""
Validate dual bridge-env routing for ROS bridge + roslibpy.

What this script checks:
1) sampler/evaluator can both use the same env_id (environment type)
   while being isolated by different bridge_env_id values.
2) under concurrent requests, each channel only receives its own responses.
3) request/response loop stays stable without cross-channel message pollution.

Prerequisites:
- rosbridge is running (default ws://localhost:9090)
- gazebo_bridge.py is running with env_id-aware protocol + lock
"""

import argparse
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import roslibpy


@dataclass
class ChannelStats:
    name: str
    sent_reset: int = 0
    sent_step: int = 0
    recv_reset: int = 0
    recv_step: int = 0
    observed_other_channel: int = 0
    invalid_env_id: int = 0
    timeouts: int = 0
    reset_latency_ms: List[float] = field(default_factory=list)
    step_latency_ms: List[float] = field(default_factory=list)


class ChannelEndpoint:
    def __init__(self, ros: roslibpy.Ros, bridge_env_id: str, timeout: float, action: np.ndarray):
        self.bridge_env_id = bridge_env_id
        self.timeout = timeout
        self.action = action.astype(np.float32)

        self._pub_action = roslibpy.Topic(ros, '/rl/action', 'std_msgs/String')
        self._pub_reset = roslibpy.Topic(ros, '/rl/reset', 'std_msgs/String')

        self._step_event = threading.Event()
        self._reset_event = threading.Event()
        self._step_payload: Optional[Dict] = None
        self._reset_payload: Optional[Dict] = None

    def on_step_result(self, msg: Dict, stats: ChannelStats):
        payload = json.loads(msg['data'])
        msg_env_id = payload.get('env_id')
        if not isinstance(msg_env_id, str):
            stats.invalid_env_id += 1
            return
        if msg_env_id != self.bridge_env_id:
            stats.observed_other_channel += 1
            return
        self._step_payload = payload
        self._step_event.set()

    def on_reset_result(self, msg: Dict, stats: ChannelStats):
        payload = json.loads(msg['data'])
        msg_env_id = payload.get('env_id')
        if not isinstance(msg_env_id, str):
            stats.invalid_env_id += 1
            return
        if msg_env_id != self.bridge_env_id:
            stats.observed_other_channel += 1
            return
        self._reset_payload = payload
        self._reset_event.set()

    def reset_once(self, n_obstacles: int, stats: ChannelStats) -> bool:
        self._reset_event.clear()
        self._reset_payload = None
        payload = {
            'env_id': self.bridge_env_id,
            'n_obstacles': int(n_obstacles),
        }

        t0 = time.perf_counter()
        self._pub_reset.publish(roslibpy.Message({'data': json.dumps(payload)}))
        stats.sent_reset += 1

        if not self._reset_event.wait(self.timeout):
            stats.timeouts += 1
            return False

        stats.recv_reset += 1
        stats.reset_latency_ms.append((time.perf_counter() - t0) * 1000.0)
        return True

    def step_once(self, stats: ChannelStats) -> bool:
        self._step_event.clear()
        self._step_payload = None
        payload = {
            'env_id': self.bridge_env_id,
            'action': self.action.tolist(),
        }

        t0 = time.perf_counter()
        self._pub_action.publish(roslibpy.Message({'data': json.dumps(payload)}))
        stats.sent_step += 1

        if not self._step_event.wait(self.timeout):
            stats.timeouts += 1
            return False

        stats.recv_step += 1
        stats.step_latency_ms.append((time.perf_counter() - t0) * 1000.0)
        return True


def _format_latency(values: List[float]) -> str:
    if not values:
        return 'n/a'
    p50 = statistics.median(values)
    p95 = np.percentile(values, 95)
    return f'count={len(values)}, p50={p50:.1f}ms, p95={p95:.1f}ms, max={max(values):.1f}ms'


def run_test(args) -> int:
    ros = roslibpy.Ros(host=args.host, port=args.port)
    ros.run()
    if not ros.is_connected:
        print('[FAIL] Could not connect to rosbridge.')
        return 2

    print(f'[INFO] Connected to rosbridge: ws://{args.host}:{args.port}')

    sampler_stats = ChannelStats(name='sampler')
    evaluator_stats = ChannelStats(name='evaluator')

    sampler_ep = ChannelEndpoint(
        ros=ros,
        bridge_env_id=args.sampler_bridge_env_id,
        timeout=args.timeout,
        action=np.array(args.sampler_action, dtype=np.float32),
    )
    evaluator_ep = ChannelEndpoint(
        ros=ros,
        bridge_env_id=args.evaluator_bridge_env_id,
        timeout=args.timeout,
        action=np.array(args.evaluator_action, dtype=np.float32),
    )

    sub_step = roslibpy.Topic(ros, '/rl/result', 'std_msgs/String')
    sub_reset = roslibpy.Topic(ros, '/rl/reset_result', 'std_msgs/String')

    sub_step.subscribe(lambda msg: (
        sampler_ep.on_step_result(msg, sampler_stats),
        evaluator_ep.on_step_result(msg, evaluator_stats),
    ))
    sub_reset.subscribe(lambda msg: (
        sampler_ep.on_reset_result(msg, sampler_stats),
        evaluator_ep.on_reset_result(msg, evaluator_stats),
    ))

    # Phase 1: reset both channels
    print('[INFO] Phase 1: reset both channels')
    ok = True
    ok &= sampler_ep.reset_once(args.n_obstacles, sampler_stats)
    ok &= evaluator_ep.reset_once(args.n_obstacles, evaluator_stats)

    # Phase 2: concurrent step traffic
    print(f'[INFO] Phase 2: concurrent step traffic, rounds={args.rounds}')
    barrier = threading.Barrier(2)

    def _worker(endpoint: ChannelEndpoint, stats: ChannelStats):
        nonlocal ok
        for _ in range(args.rounds):
            try:
                barrier.wait(timeout=args.timeout)
            except threading.BrokenBarrierError:
                stats.timeouts += 1
                ok = False
                return
            if not endpoint.step_once(stats):
                ok = False
                return

    t1 = threading.Thread(target=_worker, args=(sampler_ep, sampler_stats), daemon=True)
    t2 = threading.Thread(target=_worker, args=(evaluator_ep, evaluator_stats), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=args.timeout * max(2, args.rounds))
    t2.join(timeout=args.timeout * max(2, args.rounds))

    if t1.is_alive() or t2.is_alive():
        print('[FAIL] Worker thread did not finish (possible deadlock/timeout).')
        ok = False

    # Cleanup subscribers and ROS connection
    sub_step.unsubscribe()
    sub_reset.unsubscribe()
    ros.terminate()

    print('\n===== RESULT =====')
    for s in (sampler_stats, evaluator_stats):
        print(f'[{s.name}] sent(reset/step)=({s.sent_reset}/{s.sent_step}), '
              f'recv(reset/step)=({s.recv_reset}/{s.recv_step}), '
              f'observed_other_channel={s.observed_other_channel}, '
              f'invalid_env_id={s.invalid_env_id}, timeouts={s.timeouts}')
        print(f'[{s.name}] reset latency: {_format_latency(s.reset_latency_ms)}')
        print(f'[{s.name}] step latency : {_format_latency(s.step_latency_ms)}')

    expected_step = args.rounds
    pass_conditions = [
        ok,
        sampler_stats.recv_step == expected_step,
        evaluator_stats.recv_step == expected_step,
        sampler_stats.timeouts == 0,
        evaluator_stats.timeouts == 0,
        sampler_stats.invalid_env_id == 0,
        evaluator_stats.invalid_env_id == 0,
    ]

    if all(pass_conditions):
        print('\n[PASS] Dual bridge-env mechanism is feasible under this test load.')
        return 0

    print('\n[FAIL] Mechanism did not pass all checks. Please inspect logs and bridge node output.')
    return 1


def parse_args():
    parser = argparse.ArgumentParser(description='Validate dual bridge-env ROS communication.')
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--timeout', type=float, default=8.0)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--n_obstacles', type=int, default=3)

    parser.add_argument('--sampler_bridge_env_id', type=str, default='sampler')
    parser.add_argument('--evaluator_bridge_env_id', type=str, default='evaluator')

    parser.add_argument('--sampler_action', type=float, nargs=2, default=[0.0, 0.01])
    parser.add_argument('--evaluator_action', type=float, nargs=2, default=[0.0, -0.01])

    return parser.parse_args()


if __name__ == '__main__':
    raise SystemExit(run_test(parse_args()))
