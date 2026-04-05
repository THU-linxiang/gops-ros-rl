#!/usr/bin/env python3
"""
test_wrapper.py
测试 PythGazeboParking wrapper 是否可用
运行环境：conda gops
前提条件：
  1. Gazebo 已启动（roslaunch my_env_generator parking.launch）
  2. rosbridge 已启动（roslaunch rosbridge_server rosbridge_websocket.launch）
  3. gazebo_bridge 已启动（python3 gazebo_bridge.py）
运行方式：python3 test_wrapper.py
"""

import sys
import traceback
import numpy as np

sys.path.append('/home/lin/combined-repo/GOPS/gops/env/env_gazebo')
from pyth_gazebo_parking import PythGazeboParking


def test_connection():
    """测试 rosbridge 连接"""
    print("\n" + "="*50)
    print("测试1：rosbridge 连接")
    print("="*50)
    try:
        env = PythGazeboParking(timeout=10.0)
        print("✓ 连接成功")
        return env
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        sys.exit(1)


def test_spaces(env):
    """测试观测空间和动作空间"""
    print("\n" + "="*50)
    print("测试2：空间定义")
    print("="*50)

    print(f"观测空间: {env.observation_space}")
    print(f"  shape : {env.observation_space.shape}")   # 期望 (43,)
    print(f"动作空间: {env.action_space}")
    print(f"  low   : {env.action_space.low}")
    print(f"  high  : {env.action_space.high}")
    print(f"  shape : {env.action_space.shape}")        # 期望 (2,)

    assert env.observation_space.shape == (43,), "观测空间维度错误，期望 (43,)"
    assert env.action_space.shape == (2,),       "动作空间维度错误，期望 (2,)"
    print("✓ 空间定义正确")


def test_reset(env):
    """测试 reset"""
    print("\n" + "="*50)
    print("测试3：reset()")
    print("="*50)

    print("发送 reset 请求...")
    obs = env.reset()

    print(f"obs shape : {obs.shape}")    # 期望 (43,)
    print(f"obs dtype : {obs.dtype}")    # 期望 float32
    print(f"obs 前5维 : {obs[:5]}")
    print(f"obs 是否含 NaN : {np.any(np.isnan(obs))}")
    print(f"obs 是否含 Inf : {np.any(np.isinf(obs))}")

    assert obs.shape == (43,),      "obs 维度错误"
    assert obs.dtype == np.float32, "obs 类型错误"
    assert not np.any(np.isnan(obs)), "obs 含有 NaN"
    assert not np.any(np.isinf(obs)), "obs 含有 Inf"
    print("✓ reset 正常")
    return obs


def test_step_zero_action(env):
    """测试零动作 step"""
    print("\n" + "="*50)
    print("测试4：step() 零动作")
    print("="*50)

    action = np.array([0.0, 0.0], dtype=np.float32)
    print(f"发送 action: {action}")

    obs, reward, done, info = env.step(action)

    print(f"obs shape  : {obs.shape}")
    print(f"reward     : {reward:.4f}")
    print(f"done       : {done}")
    print(f"info       : {info}")
    print(f"obs 是否含 NaN : {np.any(np.isnan(obs))}")

    assert obs.shape == (43,), "obs 维度错误"
    assert isinstance(reward, float), "reward 类型错误"
    assert isinstance(done, bool),    "done 类型错误"
    assert isinstance(info, dict),    "info 类型错误"
    print("✓ step 零动作正常")


def test_step_random_actions(env, n_steps=10):
    """测试随机动作连续运行"""
    print("\n" + "="*50)
    print(f"测试5：连续 {n_steps} 步随机动作")
    print("="*50)

    obs = env.reset()
    total_reward = 0.0

    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"  step {i+1:2d} | action={action} | reward={reward:7.3f} | done={done}")

        if done:
            print(f"  → done=True，提前结束（step {i+1}），执行 reset")
            obs = env.reset()

    print(f"累计奖励: {total_reward:.3f}")
    print("✓ 连续 step 正常")


def test_done_and_reset(env):
    """测试 done 之后 reset 是否正常"""
    print("\n" + "="*50)
    print("测试6：done 后 reset")
    print("="*50)

    obs = env.reset()
    done = False
    steps = 0

    # 一直发正向速度，等待碰墙或超时
    while not done and steps < 300:
        action = np.array([0.03, 0.0], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        steps += 1

    print(f"done 触发于第 {steps} 步，info={info}")
    print("执行 reset...")
    obs = env.reset()
    print(f"reset 后 obs shape: {obs.shape}")
    assert obs.shape == (43,), "reset 后 obs 维度错误"
    print("✓ done 后 reset 正常")


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║       PythGazeboParking Wrapper 测试         ║")
    print("╚══════════════════════════════════════════════╝")

    env = None
    try:
        env = test_connection()
        test_spaces(env)
        test_reset(env)
        test_step_zero_action(env)
        test_step_random_actions(env, n_steps=10)
        test_done_and_reset(env)

        print("\n" + "="*50)
        print("全部测试通过 ✓")
        print("="*50)

    except AssertionError as e:
        print(f"\n✗ 断言失败: {e}")
        traceback.print_exc()

    except TimeoutError as e:
        print(f"\n✗ 超时: {e}")
        print("请检查 gazebo_bridge 是否正在运行")

    except Exception as e:
        print(f"\n✗ 未知错误: {e}")
        traceback.print_exc()

    finally:
        if env is not None:
            env.close()
            print("环境已关闭")


if __name__ == "__main__":
    main()

