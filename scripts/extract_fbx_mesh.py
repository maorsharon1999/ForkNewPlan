"""
One-time script: extracts the T-pose mesh from a binary FBX 7.x file and
writes it as models/x_bot.glb (readable by trimesh).

Usage:
    python scripts/extract_fbx_mesh.py
"""

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pygltflib

FBX_PATH = Path("models/x_bot.fbx")
GLB_PATH  = Path("models/x_bot.glb")


# ── Minimal binary FBX 7.x geometry reader ──────────────────────────────────

def _fbx_skip_prop(data, pos):
    t = data[pos]; pos += 1
    scalars = {89: 2, 67: 1, 73: 4, 70: 4, 68: 8, 76: 8}  # Y C I F D L
    if t in scalars:
        return pos + scalars[t]
    if t in (102, 100, 105, 108, 98):  # f d i l b  (array types)
        _, _, clen = struct.unpack_from('<III', data, pos)
        return pos + 12 + clen
    if t in (83, 82):  # S R
        n = struct.unpack_from('<I', data, pos)[0]
        return pos + 4 + n
    raise ValueError(f"Unknown FBX prop type byte {t} at {pos}")


def _fbx_read_typed_array(data, pos):
    """Read one FBX array property (type byte already consumed at pos-1)."""
    t = data[pos - 1]
    cnt, enc, clen = struct.unpack_from('<III', data, pos); pos += 12
    raw = data[pos: pos + clen]
    if enc == 1:
        raw = zlib.decompress(raw)
    pos += clen
    if t == 100:   # 'd' float64
        return np.frombuffer(raw, dtype='<f8').astype(float), pos
    if t == 102:   # 'f' float32
        return np.frombuffer(raw, dtype='<f4').astype(float), pos
    if t == 105:   # 'i' int32
        return np.frombuffer(raw, dtype='<i4').copy(), pos
    if t == 108:   # 'l' int64
        return np.frombuffer(raw, dtype='<i8').astype(np.int32), pos
    raise ValueError(f"Cannot read array of type byte {t}")


def _walk_nodes(data, pos, end, in_geometry, verts_out, poly_out, is64=False):
    """Recursively walk FBX node records, collecting Geometry mesh data."""
    hdr = 25 if is64 else 13  # header size (null sentinel size too)
    while pos < end:
        if pos + hdr > len(data):
            break
        if is64:
            rec_end, nprops, _ = struct.unpack_from('<QQQ', data, pos)
            pos += 24  # 3 × uint64
        else:
            rec_end, nprops, _ = struct.unpack_from('<III', data, pos)
            pos += 12  # 3 × uint32
        if rec_end == 0:
            break
        nlen = data[pos]; pos += 1
        name = data[pos: pos + nlen].decode('ascii', errors='replace')
        pos += nlen

        if in_geometry and name in ('Vertices', 'PolygonVertexIndex') and nprops >= 1:
            t = data[pos]; pos += 1
            if name == 'Vertices' and t in (100, 102):       # float array
                arr, pos = _fbx_read_typed_array(data, pos)
                if arr.size % 3 == 0:
                    verts_out.append(arr.reshape(-1, 3))
            elif name == 'PolygonVertexIndex' and t in (105, 108):  # int array
                arr, pos = _fbx_read_typed_array(data, pos)
                poly_out.append(arr)
            else:
                pos = _fbx_skip_prop(data, pos - 1)
            for _ in range(nprops - 1):
                pos = _fbx_skip_prop(data, pos)
        else:
            for _ in range(nprops):
                pos = _fbx_skip_prop(data, pos)

        child_start = pos
        if child_start < rec_end:
            _walk_nodes(data, child_start, rec_end,
                        in_geometry or name == 'Geometry',
                        verts_out, poly_out, is64)
        pos = rec_end


def _poly_to_tris(poly_indices):
    """Convert FBX PolygonVertexIndex (negative-end convention) to triangle faces."""
    faces = []
    fan = []
    for idx in poly_indices:
        if idx < 0:
            fan.append(-(idx + 1))
            for k in range(1, len(fan) - 1):
                faces.append([fan[0], fan[k], fan[k + 1]])
            fan = []
        else:
            fan.append(idx)
    return np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)


def extract_fbx_mesh(fbx_path):
    data = Path(fbx_path).read_bytes()
    assert data[:20] == b'Kaydara FBX Binary  ', "Not a binary FBX file"
    version = struct.unpack_from('<I', data, 23)[0]
    is64 = version >= 7500  # FBX 2016+ uses 64-bit record headers

    verts_list, poly_list = [], []
    _walk_nodes(data, 27, len(data), False, verts_list, poly_list, is64)

    if not verts_list:
        raise RuntimeError("No Geometry/Vertices found in FBX file")

    # X Bot may export as multiple meshes; merge them all
    if len(verts_list) == 1:
        verts = verts_list[0]
        faces = _poly_to_tris(poly_list[0])
    else:
        parts_v, parts_f = [], []
        offset = 0
        for v, p in zip(verts_list, poly_list):
            f = _poly_to_tris(p)
            parts_v.append(v)
            parts_f.append(f + offset)
            offset += len(v)
        verts = np.vstack(parts_v)
        faces = np.vstack(parts_f) if parts_f else np.zeros((0, 3), dtype=np.int32)

    return verts.astype(np.float32), faces.astype(np.uint32)


# ── Write GLB via pygltflib ─────────────────────────────────────────────────

def write_glb(verts, faces, out_path):
    v_bytes = verts.astype(np.float32).tobytes()
    f_bytes = faces.astype(np.uint32).tobytes()
    # layout: positions block first, then indices block
    buf_data = v_bytes + f_bytes

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=0),
                indices=1,
            )
        ])],
        accessors=[
            # accessor 0 = POSITION
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(verts),
                type=pygltflib.VEC3,
                max=verts.max(axis=0).tolist(),
                min=verts.min(axis=0).tolist(),
            ),
            # accessor 1 = indices
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.UNSIGNED_INT,
                count=int(faces.size),
                type=pygltflib.SCALAR,
                max=[int(faces.max())],
                min=[0],
            ),
        ],
        bufferViews=[
            # bufferView 0 = vertex positions
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(v_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ),
            # bufferView 1 = indices
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(v_bytes),
                byteLength=len(f_bytes),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
        ],
        buffers=[pygltflib.Buffer(byteLength=len(buf_data))],
    )
    gltf.set_binary_blob(buf_data)
    gltf.save(str(out_path))


if __name__ == "__main__":
    print(f"Reading {FBX_PATH} …")
    verts, faces = extract_fbx_mesh(FBX_PATH)
    print(f"  verts: {verts.shape}, faces: {faces.shape}")
    print(f"  bounds min: {verts.min(axis=0)}")
    GLB_PATH.parent.mkdir(exist_ok=True)
    write_glb(verts, faces, GLB_PATH)
    print(f"Wrote {GLB_PATH}  ({GLB_PATH.stat().st_size / 1024:.0f} kB)")
