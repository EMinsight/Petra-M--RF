#
#  subroutine for analyzing port goemetry
#
from petram.helper.geom import find_circle_center_radius
from petram.helper.geom import connect_pairs
from petram.model import Bdry
from petram.mfem_config import use_parallel

import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('RFPortGeomgetry')

if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    from petram.helper.mpi_recipes import *
else:
    import mfem.ser as mfem
    nicePrint = dprint1


def analyze_geom_norm_ref(mesh, ibe, ref):

    el = mesh.GetBdrElement(ibe[0])
    Tr = mesh.GetBdrElementTransformation(ibe[0])
    rules = mfem.IntegrationRules()
    ir = rules.Get(el.GetGeometryType(), 1)
    Tr.SetIntPoint(ir.IntPoint(0))
    nor = mfem.Vector(3)
    mfem.CalcOrtho(Tr.Jacobian(), nor)

    norm = nor.GetDataArray().copy()
    norm = norm / np.sqrt(np.sum(norm**2))

    if ref.strip() == "":
        ref_ptx = None
    else:
        iptx = mesh.extended_connectivity['vert2vert'][int(ref)]
        ref_ptx = mesh.GetVertexArray(iptx)

    return norm, ref_ptx


def analyze_rect_geom(port_model, mesh, ibe, norm):
    data = {}

    edges = np.array([mesh.GetBdrElementEdges(i)[0]
                      for i in ibe]).flatten()
    d = {}
    for x in edges:
        d[x] = x in d
    edges = [x for x in d.keys() if not d[x]]
    ivert = [mesh.GetEdgeVertices(x) for x in edges]
    ivert = connect_pairs(ivert)
    vv = np.vstack([mesh.GetVertexArray(i) for i in ivert])

    data["ctr"] = (np.max(vv, 0) + np.min(vv, 0)) / 2.0
    dprint1("Center " + list(data["ctr"]).__repr__())

    # rectangular port
    # idx = np.argsort(np.sqrt(np.sum((vv - data["ctr"])**2,1)))
    # corners = vv[idx[-4:],:]
    # since vv is cyclic I need to omit last one element here..
    idx = np.argsort(np.sqrt(np.sum((vv[:-1] - data["ctr"])**2, 1)))
    corners = vv[:-1][idx[-4:], :]
    for i in range(4):
        dprint1("Corner " + list(corners[i]).__repr__())
    tmp = np.sort(np.sqrt(np.sum((corners - corners[0, :])**2, 1)))
    data["b"] = tmp[1]
    data["a"] = tmp[2]
    tmp = np.argsort(np.sqrt(np.sum((corners - corners[0, :])**2, 1)))
    data["c"] = corners[0]  # corner
    data["b_vec"] = corners[tmp[1]] - corners[0]
    data["a_vec"] = np.cross(data["b_vec"], norm)
    #            data["a"]_vec = corners[tmp[2]]-corners[0]
    data["b_vec"] = data["b_vec"] / np.sqrt(np.sum(data["b_vec"]**2))
    data["a_vec"] = data["a_vec"] / np.sqrt(np.sum(data["a_vec"]**2))
    if np.sum(np.cross(data["a_vec"], data["b_vec"]) * norm) > 0:
        data["a_vec"] = -data["a_vec"]

    if port_model.mode == 'TEM':
        '''
        special handling
        set a vector along PEC-like edge, regardless the actual
        length of edges
        '''
        for i in range(nbe):
            if (edges[0] in mesh.GetBdrElementEdges(i)[0] and
                    port_model._sel_index[0] != mesh.GetBdrAttribute(i)):
                dprint1("Checking surface :", mesh.GetBdrAttribute(i))
                attr = mesh.GetBdrAttribute(i)
                break
        for node in port_model.get_root_phys().walk():
            if not isinstance(node, Bdry):
                continue
            if not node.enabled:
                continue
            if attr in node._sel_index:
                break
        from petram.model import Pair
        ivert = mesh.GetEdgeVertices(edges[0])
        vect = mesh.GetVertexArray(
            ivert[0]) - mesh.GetVertexArray(ivert[1])
        vect = vect / np.sqrt(np.sum(vect**2))
        do_swap = False
        if (isinstance(node, Pair) and
                np.abs(np.sum(data["a_vec"] * vect)) > 0.9):
            do_swap = True
        if (not isinstance(node, Pair) and
                np.abs(np.sum(data["a_vec"] * vect)) < 0.001):
            do_swap = True
        if do_swap:
            dprint1("swapping port edges")
            tmp = data["a_vec"]
            data["a_vec"] = -data["b_vec"]
            data["b_vec"] = tmp
            # - sign is to keep a \times b direction.
            tmp = data["a"]
            data["a"] = data["b"]
            data["b"] = tmp
    if data["a_vec"][np.argmax(np.abs(data["a_vec"]))] < 0:
        data["a_vec"] = -data["a_vec"]
        data["b_vec"] = -data["b_vec"]
    dprint1("Long Edge  " + data["a"].__repr__())
    dprint1("Long Edge Vec." + list(data["a_vec"]).__repr__())
    dprint1("Short Edge  " + data["b"].__repr__())
    dprint1("Short Edge Vec." + list(data["b_vec"]).__repr__())

    return data, vv


def analyze_coax_geom(mesh, ibe, norm, ref_ptx):
    data = {}

    edges = np.array([mesh.GetBdrElementEdges(i)[0]
                      for i in ibe]).flatten()
    d = {}
    for x in edges:
        d[x] = x in d
    edges = [x for x in d.keys() if not d[x]]
    ivert = [mesh.GetEdgeVertices(x) for x in edges]
    iv1, iv2 = connect_pairs(ivert)  # index of outer/inner circles
    vv1 = np.vstack([mesh.GetVertexArray(i) for i in iv1])
    vv2 = np.vstack([mesh.GetVertexArray(i) for i in iv2])
    ctr1, a1 = find_circle_center_radius(vv1, norm)
    ctr2, b1 = find_circle_center_radius(vv2, norm)
    data["ctr"] = np.mean((ctr1, ctr2), 0)
    data["a"] = a1 if a1 < b1 else b1
    data["b"] = a1 if a1 > b1 else b1

    ax1 = ref_ptx - data["ctr"]
    ax1 = ax1/np.sqrt(np.sum(ax1**2))
    ax2 = np.cross(norm, data["ctr"])
    data["ax1"] = ax1
    data["ax2"] = ax2

    dprint1("Big R:  " + data["b"].__repr__())
    dprint1("Small R: " + data["a"].__repr__())
    dprint1("Center:  " + data["ctr"].__repr__())
    dprint1("Axis1:  " + data["ax1"].__repr__())
    dprint1("Axis2:  " + data["ax2"].__repr__())
    return data, vv1


def analyze_circular_geom(mesh, ibe, norm, ref_ptx):
    data = {}

    edges = np.array([mesh.GetBdrElementEdges(i)[0]
                      for i in ibe]).flatten()
    d = {}
    for x in edges:
        d[x] = x in d
    edges = [x for x in d.keys() if not d[x]]
    ivert = [mesh.GetEdgeVertices(x) for x in edges]
    iv1 = connect_pairs(ivert)  # index of outer/inner circles
    vv1 = np.vstack([mesh.GetVertexArray(i) for i in iv1])
    ctr, a1 = find_circle_center_radius(vv1, norm)
    data["ctr"] = ctr
    data["a"] = a1

    ax1 = ref_ptx - data["ctr"]
    ax1 = ax1/np.sqrt(np.sum(ax1**2))
    ax2 = np.cross(norm, ax1)
    data["ax1"] = ax1
    data["ax2"] = ax2

    dprint1("Radius:  " + data["a"].__repr__())
    dprint1("Center:  " + data["ctr"].__repr__())
    dprint1("Axis1:  " + data["ax1"].__repr__())
    dprint1("Axis2:  " + data["ax2"].__repr__())

    return data, vv1
