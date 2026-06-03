**Quickstart**

```
uv run main.py train    # resume checkpoint, resume LR schedule
uv run main.py warm     # resume checkpoint, restart LR schedule
uv run main.py eval     # evaluate checkpoint
```

**Theory**

- [MGN Summary](https://obdwinston.github.io/post.html?slug=2026-06-03-mgn-nutshell)
- [GNN Summary](https://obdwinston.github.io/post.html?slug=2026-05-30-gnn-nutshell)

**Data**

- [Train Trajectories](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord)
- [Test Trajectories](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord)
- [Metadata](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/meta.json)

[**Weights**](checkpoint.pt)

| Phase | Trajectories | Steps     | LR Schedule |
| ----- | ------------ | --------- | ----------- |
| 1     | 100          | 200,000   | 1e-4 → 1e-6 |
| 2     | 1,000        | 1,000,000 | 1e-4 → 1e-6 |

**Verification**

Rollout (Autoregressive Prediction)

<img alt="rollout" src="https://github.com/user-attachments/assets/4ba5085d-4417-4238-bd4d-a54de8cc04d2" />

One Step (Single-Frame Prediction)

<img alt="onestep" src="https://github.com/user-attachments/assets/5347adb6-9e74-49cd-904e-fd5cb4ed48cb" />
