import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from model import NODE_TYPES

BASE = "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow"
DATA_DIR = Path(__file__).parent / "data"
NP_DTYPE = {"int32": "<i4", "float32": "<f4"}


def _read_varint(buf, pos):
    shift = result = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _iter_fields(buf):
    pos, end = 0, len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value, pos = buf[pos : pos + length], pos + length
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, value


def _parse_example(buf):
    fields = {}
    for example_field, features in _iter_fields(buf):
        if example_field != 1:
            continue
        for feature_field, entry in _iter_fields(features):
            if feature_field != 1:
                continue
            name, feature = None, None
            for entry_field, value in _iter_fields(entry):
                if entry_field == 1:
                    name = bytes(value).decode()
                elif entry_field == 2:
                    feature = value
            blobs = []
            for feature_kind, bytes_list in _iter_fields(feature):
                if feature_kind != 1:
                    continue
                for value_field, blob in _iter_fields(bytes_list):
                    if value_field == 1:
                        blobs.append(bytes(blob))
            fields[name] = b"".join(blobs)
    return fields


def _decode(fields, meta):
    decoded = {}
    for name, spec in meta["features"].items():
        array = np.frombuffer(fields[name], dtype=NP_DTYPE[spec["dtype"]])
        known = int(np.prod([d for d in spec["shape"] if d != -1]))
        shape = [array.size // known if d == -1 else d for d in spec["shape"]]
        decoded[name] = torch.from_numpy(array.reshape(shape).copy())
    return decoded


def _read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _stream_examples(url, n):
    with urllib.request.urlopen(url) as stream:
        for i in range(n):
            length = struct.unpack("<Q", _read_exact(stream, 12)[:8])[0]
            example = _read_exact(stream, length)
            _read_exact(stream, 4)
            print(f"\rdownloading trajectory {i + 1}/{n}", end="", flush=True)
            yield example
        print()


def load_meta():
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "meta.json"
    if not path.exists():
        urllib.request.urlretrieve(f"{BASE}/meta.json", path)
    return json.loads(path.read_text())


def _cache_size(path):
    return int(path.stem.split("_")[-1])


def load_trajectories(n=1, split="train"):
    DATA_DIR.mkdir(exist_ok=True)
    caches = sorted(DATA_DIR.glob(f"{split}_*.pt"), key=_cache_size)
    for cache in caches:
        if _cache_size(cache) >= n:
            return torch.load(cache, weights_only=False)[:n]
    meta = load_meta()
    trajectories = [
        _decode(_parse_example(memoryview(e)), meta)
        for e in _stream_examples(f"{BASE}/{split}.tfrecord", n)
    ]
    cache = DATA_DIR / f"{split}_{n}.pt"
    print(f"caching {cache}")
    torch.save(trajectories, cache)
    return trajectories


def triangles_to_edges(cells):
    edges = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    edges = torch.unique(edges, dim=0)
    senders, receivers = edges[:, 0], edges[:, 1]
    return torch.stack(
        [torch.cat([senders, receivers]), torch.cat([receivers, senders])]
    )


def build_static(trajectory):
    cells = trajectory["cells"][0].long()
    mesh_pos = trajectory["mesh_pos"][0].float()
    node_type = trajectory["node_type"][0].squeeze(-1).long()
    one_hot = F.one_hot(node_type, NODE_TYPES).float()
    edge_index = triangles_to_edges(cells)
    relative = mesh_pos[edge_index[0]] - mesh_pos[edge_index[1]]
    edge_attr = torch.cat([relative, relative.norm(dim=1, keepdim=True)], dim=1)
    return edge_index, edge_attr, one_hot, mesh_pos, cells


def make_graph(velocity, t, dt, edge_index, edge_attr, one_hot, mesh_pos, cells):
    return Data(
        x=torch.cat([velocity[t], one_hot], dim=1),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=(velocity[t + 1] - velocity[t]) / dt,
        mesh_pos=mesh_pos,
        cells=cells,
    )


def build_graphs(trajectory, dt):
    statics = build_static(trajectory)
    velocity = trajectory["velocity"].float()
    return [make_graph(velocity, t, dt, *statics) for t in range(velocity.shape[0] - 1)]


class CylinderDataset(Dataset):
    def __init__(self, trajectories, dt):
        self.dt = dt
        self.velocity = []
        self.statics = []
        self.index = []
        for i, trajectory in enumerate(trajectories):
            velocity = trajectory["velocity"].float()
            self.velocity.append(velocity)
            self.statics.append(build_static(trajectory))
            self.index += [(i, t) for t in range(velocity.shape[0] - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        return make_graph(self.velocity[i], t, self.dt, *self.statics[i])
