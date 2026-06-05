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
DIV_PCTL = 99


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
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, (lr_final / lr) ** (1 / steps)
    )
    step = 0
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(state["model"])
        if warm_start:
            print(f"warm start from step {state['step']} with fresh schedule")
        else:
            optimiser.load_state_dict(state["optimiser"])
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
            optimiser.zero_grad()
            loss = model.loss(data)
            loss.backward()
            optimiser.step()
            scheduler.step()
            step += 1
            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {step:7d}  loss {loss.item():.4e}  lr {lr_now:.2e}")
            if step % save_every == 0:
                save_checkpoint(model, optimiser, scheduler, step)
    save_checkpoint(model, optimiser, scheduler, step)
    return model


def save_checkpoint(model, optimiser, scheduler, step):
    tmp = CHECKPOINT.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
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


def cell_gradients(mesh_pos, cells):
    p = mesh_pos[cells]
    x, y = p[..., 0], p[..., 1]
    x0, x1, x2 = x.unbind(-1)
    y0, y1, y2 = y.unbind(-1)
    two_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    b = torch.stack([y1 - y2, y2 - y0, y0 - y1], dim=-1)
    c = torch.stack([x2 - x1, x0 - x2, x1 - x0], dim=-1)
    return b, c, two_area


def cell_divergence(velocity, mesh_pos, cells):
    b, c, two_area = cell_gradients(mesh_pos, cells)
    vel = velocity[..., cells, :]
    u, v = vel[..., 0], vel[..., 1]
    return (u * b + v * c).sum(-1) / two_area


def divergence_rms(velocity, mesh_pos, cells):
    div = cell_divergence(velocity, mesh_pos, cells)
    _, _, two_area = cell_gradients(mesh_pos, cells)
    area = two_area.abs() / 2
    return ((area * div.pow(2)).sum(-1) / area.sum()).sqrt()


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
    div_pctl=DIV_PCTL,
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
    true_div = cell_divergence(truth[::skip], mesh_pos, cells).abs()
    pred_div = cell_divergence(pred[::skip], mesh_pos, cells).abs()
    div_error = (pred_div - true_div).abs()
    div_max = torch.quantile(pred_div, div_pctl / 100).item()
    div_error_max = torch.quantile(div_error, div_pctl / 100).item()
    true_speed, pred_speed, error = (
        true_speed.numpy(),
        pred_speed.numpy(),
        error.numpy(),
    )
    true_div, pred_div, div_error = true_div.numpy(), pred_div.numpy(), div_error.numpy()

    vmax = true_speed.max()
    panels = (
        (r"actual $\|\mathbf{u}\|$", true_speed, "viridis", vmax, "gouraud"),
        (r"predicted $\|\mathbf{u}\|$", pred_speed, "viridis", vmax, "gouraud"),
        (r"percent error $\|\mathbf{u}\|$", error, "magma", error_max, "gouraud"),
        (r"actual $\nabla\cdot\mathbf{u}$", true_div, "inferno", div_max, "flat"),
        (r"predicted $\nabla\cdot\mathbf{u}$", pred_div, "inferno", div_max, "flat"),
        (r"absolute error $\nabla\cdot\mathbf{u}$", div_error, "magma", div_error_max, "flat"),
    )
    span = mesh_pos.max(0).values - mesh_pos.min(0).values
    aspect = (span[0] / span[1]).item()
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(8, 8 / aspect * len(panels)), constrained_layout=True
    )
    meshes = []
    for axis, (title, field, cmap, hi, shading) in zip(axes, panels):
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title)
        mesh = axis.tripcolor(
            triangulation, field[0], shading=shading, cmap=cmap, vmin=0, vmax=hi
        )
        figure.colorbar(mesh, ax=axis, fraction=0.025, pad=0.01)
        meshes.append(mesh)

    def update(step):
        for mesh, (_, field, *_) in zip(meshes, panels):
            mesh.set_array(field[step])
        figure.suptitle(f"step {step * skip}")

    animation.FuncAnimation(figure, update, frames=len(true_speed), blit=False).save(
        save_path, writer="ffmpeg", fps=fps
    )
    plt.close(figure)
    print(f"saved {save_path}")


def evaluate(predict, label, n_traj=TEST_TRAJ, dt=DT, device="cpu", animate=True):
    model, step = load_model(device)
    trajectories = load_trajectories(n_traj, split="test")
    drifts = []
    pred_divs = []
    true_divs = []
    for i, trajectory in enumerate(trajectories):
        graphs = build_graphs(trajectory, dt)
        free = model.free_mask(graphs[0])
        mesh_pos, cells = graphs[0].mesh_pos, graphs[0].cells
        truth = true_trajectory(graphs, dt)
        pred = predict(model, graphs, dt, device)
        drifts.append(step_rmse(pred, truth, free))
        pred_divs.append(divergence_rms(pred, mesh_pos, cells))
        true_divs.append(divergence_rms(truth, mesh_pos, cells))
        if i == 0:
            first = (graphs[0], truth, pred)
        print(
            f"{label} trajectory {i + 1:2d}/{n_traj}  rmse {drifts[-1][-1].item():.4e}"
        )
    drift = torch.stack(drifts).mean(0)
    pred_div = torch.stack(pred_divs).mean(0)
    true_div = torch.stack(true_divs).mean(0)
    print(f"checkpoint step {step}  test trajectories {n_traj}")
    for s in ROLLOUT_STEPS:
        print(
            f"{label} step {s:4d}  velocity rmse {drift[s].item():.4e}"
            f"  divergence pred {pred_div[s].item():.4e}  true {true_div[s].item():.4e}"
        )
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
    if mode == "train":
        main()
    elif mode == "warm":
        main(warm_start=True)
    elif mode == "eval":
        evaluate(onestep, "onestep")
        evaluate(rollout, "rollout")
