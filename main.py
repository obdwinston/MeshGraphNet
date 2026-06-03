import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data import CylinderDataset, build_graphs, load_trajectories
from model import MeshGraphNet

CHECKPOINT = Path("checkpoint.pt")
DT = 0.01

TRAIN_TRAJ = 1_000
STEPS = 1_000_000
BATCH_SIZE = 2
LR = 1e-4
LR_FINAL = 1e-6
NOISE_STD = 2e-2
LOG_EVERY = 10
SAVE_EVERY = 1000

TEST_TRAJ = 10
ROLLOUT_STEPS = (1, 50, 200, 500, 599)

FPS = 15
SKIP = 3
EPS = 1e-2
ERROR_MAX = 100.0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(
    model,
    dataset,
    steps=STEPS,
    batch_size=BATCH_SIZE,
    lr=LR,
    lr_final=LR_FINAL,
    noise_std=NOISE_STD,
    dt=DT,
    log_every=LOG_EVERY,
    save_every=SAVE_EVERY,
    warm_start=False,
    device="cpu",
):
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, (lr_final / lr) ** (1 / steps)
    )
    step = 0
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(state["model"])
        if warm_start:
            print(f"warm start from step {state['step']} with fresh schedule")
        else:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = state["step"]
            print(f"resumed at step {step}")
    target = step + steps
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while step < target:
        for data in loader:
            if step >= target:
                break
            data = data.to(device)
            free = model.free_mask(data)
            noise = torch.randn_like(data.x[:, :2]) * noise_std
            noise[~free] = 0
            data.x[:, :2] += noise
            data.y -= noise / dt
            optimizer.zero_grad()
            loss = model.loss(data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {step:7d}  loss {loss.item():.4e}  lr {lr_now:.2e}")
            if step % save_every == 0:
                save_checkpoint(model, optimizer, scheduler, step)
    save_checkpoint(model, optimizer, scheduler, step)
    return model


def save_checkpoint(model, optimizer, scheduler, step):
    tmp = CHECKPOINT.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        tmp,
    )
    tmp.replace(CHECKPOINT)


def load_model(device="cpu"):
    state = torch.load(CHECKPOINT, map_location=device)
    model = MeshGraphNet()
    model.load_state_dict(state["model"])
    model.eval().to(device)
    return model, state["step"]


@torch.no_grad()
def onestep(model, graphs, dt=DT, device="cpu"):
    model.eval().to(device)
    trajectory = [graphs[0].x[:, :2].clone()]
    for data in graphs:
        data = data.to(device)
        free = model.free_mask(data)
        velocity = data.x[:, :2].clone()
        velocity[free] += model.out_norm.inverse(model(data))[free] * dt
        trajectory.append(velocity.cpu().clone())
    return torch.stack(trajectory)


@torch.no_grad()
def rollout(model, graphs, dt=DT, device="cpu"):
    model.eval().to(device)
    graph = graphs[0].clone().to(device)
    free = model.free_mask(graph)
    velocity = graph.x[:, :2].clone()
    trajectory = [velocity.cpu().clone()]
    for _ in graphs:
        velocity[free] += model.out_norm.inverse(model(graph))[free] * dt
        graph.x[:, :2] = velocity
        trajectory.append(velocity.cpu().clone())
    return torch.stack(trajectory)


def true_trajectory(graphs, dt):
    frames = [g.x[:, :2] for g in graphs]
    frames.append(graphs[-1].x[:, :2] + graphs[-1].y * dt)
    return torch.stack(frames)


def step_rmse(pred, truth, free):
    return (pred[:, free] - truth[:, free]).pow(2).sum(2).mean(1).sqrt()


def animate_compare(
    mesh_pos,
    cells,
    truth,
    pred,
    save_path,
    fps=FPS,
    skip=SKIP,
    eps=EPS,
    error_max=ERROR_MAX,
):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    position = mesh_pos.cpu().numpy()
    triangulation = Triangulation(position[:, 0], position[:, 1], cells.cpu().numpy())
    true_speed = truth[::skip].norm(dim=2)
    pred_speed = pred[::skip].norm(dim=2)
    error = (100 * (pred_speed - true_speed).abs() / (true_speed + eps)).clamp(
        max=error_max
    )
    true_speed, pred_speed, error = (
        true_speed.numpy(),
        pred_speed.numpy(),
        error.numpy(),
    )

    vmax = true_speed.max()
    panels = (
        ("actual", true_speed, "viridis", vmax),
        ("predicted", pred_speed, "viridis", vmax),
        ("error %", error, "magma", error_max),
    )
    figure, axes = plt.subplots(3, 1, figsize=(8, 7.5))
    meshes = []
    for axis, (title, field, cmap, hi) in zip(axes, panels):
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title)
        mesh = axis.tripcolor(
            triangulation, field[0], shading="gouraud", cmap=cmap, vmin=0, vmax=hi
        )
        figure.colorbar(mesh, ax=axis)
        meshes.append(mesh)

    def update(step):
        for mesh, (_, field, _, _) in zip(meshes, panels):
            mesh.set_array(field[step])
        figure.suptitle(f"step {step * skip}")
        return meshes

    animation.FuncAnimation(figure, update, frames=len(true_speed), blit=False).save(
        save_path, writer="ffmpeg", fps=fps
    )
    plt.close(figure)
    print(f"saved {save_path}")


def evaluate(predict, label, n_traj=TEST_TRAJ, dt=DT, device="cpu", animate=True):
    model, step = load_model(device)
    trajectories = load_trajectories(n_traj, split="test")
    drifts = []
    for i, trajectory in enumerate(trajectories):
        graphs = build_graphs(trajectory, dt)
        free = model.free_mask(graphs[0])
        truth = true_trajectory(graphs, dt)
        pred = predict(model, graphs, dt, device)
        drifts.append(step_rmse(pred, truth, free))
        if i == 0:
            first = (graphs[0], truth, pred)
        print(
            f"{label} trajectory {i + 1:2d}/{n_traj}  rmse {drifts[-1][-1].item():.4e}"
        )
    drift = torch.stack(drifts).mean(0)
    print(f"checkpoint step {step}  test trajectories {n_traj}")
    for s in ROLLOUT_STEPS:
        print(f"{label} step {s:4d}  velocity rmse {drift[s].item():.4e}")
    if animate:
        init, truth, pred = first
        animate_compare(init.mesh_pos, init.cells, truth, pred, f"test_{label}.mp4")


def main(n_traj=TRAIN_TRAJ, dt=DT, warm_start=False):
    device = get_device()
    print(f"device {device}")
    trajectories = load_trajectories(n_traj)
    dataset = CylinderDataset(trajectories, dt)
    train(MeshGraphNet(), dataset, dt=dt, warm_start=warm_start, device=device)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":  # resume checkpoint, resume LR schedule
        main()
    elif mode == "warm":  # resume checkpoint, restart LR schedule
        main(warm_start=True)
    elif mode == "eval":  # evaluate checkpoint
        evaluate(onestep, "onestep")
        evaluate(rollout, "rollout")
