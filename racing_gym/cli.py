"""Command line interface: ``racing-gym <command>`` (or ``python -m racing_gym <command>``)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import EnvConfig, load_config


def _env_cfg(args) -> EnvConfig:
    cfg = load_config(args.config, args.set).env if getattr(args, "config", None) else load_config(None, args.set).env
    if getattr(args, "track", None):
        cfg = replace(cfg, track=args.track)
    if getattr(args, "car", None):
        cfg = replace(cfg, car=args.car)
    return cfg


def cmd_train(args):
    from .training import train

    cfg = load_config(args.config, args.set)
    if args.timesteps:
        cfg = replace(cfg, train=replace(cfg.train, total_timesteps=args.timesteps))
    seeds = args.seeds if args.seeds is not None else list(cfg.train.seeds)
    for seed in seeds:
        run_dir = train(cfg, seed=seed, out_dir=args.out)
        print(f"done: {run_dir}")


def cmd_evaluate(args):
    from .evaluation import make_policy, run_episodes, save_json, summarize

    env_cfg = _env_cfg(args)
    tracks = args.tracks or [env_cfg.track]
    rows = []
    for spec in args.policies:
        policy, pol_env, label = make_policy(spec, env_cfg, seed=args.seed, algo=args.algo,
                                             deterministic=not args.stochastic)
        res = run_episodes(policy, pol_env, tracks, episodes=args.episodes, seed=10_000 + args.seed,
                           render_mode="human" if args.render else None)
        rows.append({"policy": label, **summarize(res), "episodes_detail": res})
    print(f"\ntracks: {', '.join(tracks)}   episodes per track: {args.episodes}\n")
    print("| policy | return | laps | lap fraction | crash rate | avg speed |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['policy']} | {r['return_mean']:.1f} ± {r['return_std']:.1f} | {r['laps_mean']:.2f} | "
              f"{r['lap_fraction_mean']:.2f} | {r['crash_rate']:.0%} | {r['avg_speed']:.2f} |")
    if args.json:
        save_json(rows, args.json)


def cmd_play(args):
    import pygame

    from .env import RaceEnv

    env = RaceEnv(replace(_env_cfg(args), random_start=False, max_steps=10**9, stall_steps=10**9),
                  render_mode="human")
    obs, _ = env.reset(seed=0)
    print("arrows: drive   R: reset   Esc: quit")
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break
        if keys[pygame.K_r]:
            env.reset()
        steer = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 1.0
        throttle = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 1.0
        if env.config.action_type == "discrete":
            from .env import DISCRETE_ACTIONS

            action = DISCRETE_ACTIONS.index((steer, throttle))
        else:
            action = [steer, throttle]
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            st = info["episode_stats"]
            print(f"episode over: laps={st['laps']} return={sum(st['reward_terms'].values()):.1f}")
            env.reset()
    env.close()


def cmd_watch(args):
    from .evaluation import make_policy, run_episodes, summarize

    env_cfg = replace(_env_cfg(args), random_start=False)
    policy, pol_env, label = make_policy(args.policy, env_cfg, algo=args.algo, deterministic=not args.stochastic)
    res = run_episodes(policy, pol_env, [env_cfg.track], episodes=args.episodes, render_mode="human")
    print(label, summarize(res))


def cmd_record(args):
    from .gif import record_gif

    env_cfg = replace(_env_cfg(args), random_start=False, max_steps=args.steps, stall_steps=args.steps)
    out = record_gif(args.policies, env_cfg, args.out, steps=args.steps, algo=args.algo, every=args.every,
                     scale=args.scale)
    print(f"wrote {out}")


def cmd_export(args):
    from .export import export_policy

    out = export_policy(args.run, args.out, name=args.name, description=args.description, stages=args.stages)
    print(f"wrote {out}")


def cmd_plot(args):
    from .plotting import plot_runs

    out = plot_runs(args.runs, args.out, metric=args.metric, title=args.title)
    print(f"wrote {out}")


def cmd_tracks(args):
    from .plotting import plot_tracks

    out = plot_tracks(args.out, args.tracks)
    print(f"wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="racing-gym", description="Train, watch and compare racing RL agents.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, config=True):
        if config:
            sp.add_argument("--config", "-c", help="TOML experiment config (see configs/)")
        sp.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. -s env.reward.crash=-50 (repeatable)")
        sp.add_argument("--track", help="preset name or random:<seed>")
        sp.add_argument("--car", help="car preset: balanced, grippy, rocket")

    sp = sub.add_parser("train", help="train an agent")
    sp.add_argument("--config", "-c", required=True)
    sp.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE")
    sp.add_argument("--seeds", type=int, nargs="+", help="override train.seeds")
    sp.add_argument("--timesteps", type=int, help="override train.total_timesteps")
    sp.add_argument("--out", default="runs")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="compare policies on the same seeded episodes")
    common(sp)
    sp.add_argument("policies", nargs="+", help="run directories, or: random, heuristic, untrained")
    sp.add_argument("--tracks", nargs="+", help="evaluate on these tracks (default: config track)")
    sp.add_argument("--episodes", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--algo", default="ppo", help="algorithm for the 'untrained' baseline")
    sp.add_argument("--stochastic", action="store_true")
    sp.add_argument("--render", action="store_true")
    sp.add_argument("--json", help="also write the results to this JSON file")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("play", help="drive yourself with the arrow keys")
    common(sp)
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("watch", help="watch a policy drive")
    common(sp)
    sp.add_argument("policy", help="run directory, or: random, heuristic, untrained")
    sp.add_argument("--episodes", type=int, default=3)
    sp.add_argument("--algo", default="ppo")
    sp.add_argument("--stochastic", action="store_true")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("record", help="record a GIF (several policies side by side)")
    common(sp)
    sp.add_argument("policies", nargs="+")
    sp.add_argument("--out", default="race.gif")
    sp.add_argument("--steps", type=int, default=600)
    sp.add_argument("--every", type=int, default=2, help="keep every n-th frame")
    sp.add_argument("--scale", type=float, default=0.5)
    sp.add_argument("--algo", default="ppo")
    sp.set_defaults(func=cmd_record)

    sp = sub.add_parser("export", help="export a trained policy for the browser playground")
    sp.add_argument("run")
    sp.add_argument("--out", required=True, help="e.g. web/models/my_agent.js")
    sp.add_argument("--name")
    sp.add_argument("--description", default="")
    sp.add_argument("--stages", type=int, default=1,
                    help="also export N-1 earlier checkpoints so the playground can replay learning")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("plot", help="plot learning curves of one or more experiments")
    sp.add_argument("runs", nargs="+", help="experiment directories (runs/<name>), each with seed*/ inside")
    sp.add_argument("--out", default="learning_curves.png")
    sp.add_argument("--metric", default="return", choices=["return", "laps", "crash_rate", "avg_speed"])
    sp.add_argument("--title", default="")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("tracks", help="draw the built-in tracks")
    sp.add_argument("tracks", nargs="*")
    sp.add_argument("--out", default="tracks.png")
    sp.set_defaults(func=cmd_tracks)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
