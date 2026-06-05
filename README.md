### Quickstart

```
uv run main.py train    # resume checkpoint, resume LR schedule
uv run main.py warm     # resume checkpoint, restart LR schedule
uv run main.py eval     # evaluate checkpoint
```

### Theory

- [MGN Summary](https://obdwinston.github.io/post.html?slug=2026-06-03-mgn-nutshell)
- [GNN Summary](https://obdwinston.github.io/post.html?slug=2026-05-30-gnn-nutshell)

### Data

- [Train Trajectories](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord)
- [Test Trajectories](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord)
- [Metadata](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/meta.json)

### [Weights](checkpoint.pt)

| Phase | Trajectories | Gradient Steps | LR Schedule |
| ----- | ------------ | -------------- | ----------- |
| 1     | 100          | 200,000        | 1e-4 → 1e-6 |
| 2     | 1,000        | 1,000,000      | 1e-4 → 1e-6 |

### Verification

#### Rollout (Autoregressive Prediction)

https://github.com/user-attachments/assets/8e44ba40-6068-496f-ad2e-9317003dd36f

#### One Step (Single-Frame Prediction)

https://github.com/user-attachments/assets/80468f11-50b7-4579-b2a7-553eeb9cc9cf

### References

- [Paper](https://arxiv.org/abs/2010.03409)
- [Video](https://iclr.cc/virtual/2021/poster/2837)
- [Code](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
- [Website](https://sites.google.com/view/meshgraphnets)
