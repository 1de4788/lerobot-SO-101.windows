"""Smoke test: verify CUDA GPU + lerobot + pusht env all work on this machine."""
import time

import torch

from lerobot.envs import make_env, make_env_config


def main():
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda version:", torch.version.cuda)

    t0 = time.time()
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"cuda matmul 2048x2048: {time.time() - t0:.3f}s")

    cfg = make_env_config("pusht")
    envs = make_env(cfg, n_envs=1, use_async_envs=False)
    print("env suites:", list(envs.keys()))
    suite = list(envs.keys())[0]
    env = envs[suite][0]
    print("env created:", type(env).__name__)

    obs, info = env.reset()
    print("reset ok, obs keys:", list(obs.keys()))
    a = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(a)
    print("step ok, reward:", rew)
    env.close()
    print("SMOKE ENV OK")


if __name__ == "__main__":
    main()
