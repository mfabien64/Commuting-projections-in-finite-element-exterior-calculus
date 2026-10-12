
"""Lowest-order Pi_0^1 projector for straight-sided tetrahedral NGSolve meshes.

Public use:
    P = Pi1Projector(mesh)
    Pi1_u = P(u)

Normal public application mode constructs only the operator data needed for
Pi_0^1. It does not run convergence studies or verification/error tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from netgen.csg import Pnt
from netgen.meshing import Element2D, Element3D, FaceDescriptor, Mesh as NGMesh, MeshPoint
from ngsolve import BND, EDGE, FACE, VERTEX, BilinearForm, CoefficientFunction, Cross, ElementId, GridFunction, H1, HCurl, HDiv, Integrate, InnerProduct, L2, LinearForm, Mesh, NodeId, NumberSpace, VOL, curl, div, dx, grad, specialcf, x, y, z
Array = np.ndarray
Edge = Tuple[int, int]
Face = Tuple[int, int, int]

@dataclass
class TetComplex:
    coords: Array
    tets: Array
    parent: Array | None = None
    n_original_vertices: int | None = None
    bary_parent: Dict[int, int] | None = None

@dataclass
class Patch:
    mesh: Mesh
    coords: Array
    tets: Array
    local_to_global_nodes: Array
    global_to_local_nodes: Dict[int, int]
    local_to_global_elements: Array
    parent_original_tet: Array

@dataclass
class Z0Weight:
    vertex: int
    values_global_alfeld_cells: Array
    reproduction_error: float
    integral_error: float
    zero_mean_psi_error: float

@dataclass
class Z1Result:
    edge: Edge
    original_patch_tets: Array
    alfeld_patch_cells: Array
    fine_patch: Patch
    coarse_patch: Patch
    mu: GridFunction
    rhs_z0_boundary: GridFunction
    eta1: GridFunction
    multiplier_u: GridFunction
    psi1_fine: GridFunction
    z1: GridFunction
    eta_div_error: float
    eta_boundary_flux_norm: float
    eta_mean_multiplier: float
    eta_stability_ratio: float
    psi_equation_error: float
    psi_kernel_compatibility: float
    psi_kernel_orthogonality: float
    z1_div_error: float
    z1_boundary_flux_norm: float
    reproduction_error: float

def oriented_tets(coords: Array, tets: Array) -> Array:
    out = np.asarray(tets, dtype=int).copy()
    for k, tet in enumerate(out):
        x0, x1, x2, x3 = coords[tet]
        det = np.linalg.det(np.column_stack((x1 - x0, x2 - x0, x3 - x0)))
        if abs(det) < 1e-14:
            raise ValueError(f'Degenerate tetrahedron {k}')
        if det < 0:
            out[k, [0, 1]] = out[k, [1, 0]]
    return out

def canonical_edges(tets: Array) -> List[Edge]:
    edges = set()
    for tet in np.asarray(tets, dtype=int):
        for i, j in combinations(map(int, tet), 2):
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)

def boundary_faces(tets: Array) -> List[Face]:
    count: Dict[Face, int] = {}
    representative: Dict[Face, Face] = {}
    local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    for tet in np.asarray(tets, dtype=int):
        for loc in local_faces:
            oriented = tuple((int(tet[i]) for i in loc))
            key = tuple(sorted(oriented))
            count[key] = count.get(key, 0) + 1
            representative[key] = oriented
    return [representative[key] for key, c in count.items() if c == 1]

def make_ngsolve_tet_mesh(coords: Array, tets: Array, boundary_name: str='patch_boundary') -> Mesh:
    coords = np.asarray(coords, dtype=float)
    tets = oriented_tets(coords, np.asarray(tets, dtype=int))
    ngm = NGMesh()
    ngm.dim = 3
    pids = [ngm.Add(MeshPoint(Pnt(*map(float, p)))) for p in coords]
    ngm.SetMaterial(1, 'domain')
    for tet in tets:
        ngm.Add(Element3D(1, [pids[int(i)] for i in tet]))
    ngm.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    for face in boundary_faces(tets):
        ngm.Add(Element2D(1, [pids[int(i)] for i in face]))
    ngm.SetBCName(0, boundary_name)
    ngm.Compress()
    return Mesh(ngm)

def alfeld_split(original: TetComplex) -> TetComplex:
    coords = [p.copy() for p in np.asarray(original.coords, dtype=float)]
    subtets: List[List[int]] = []
    parent: List[int] = []
    bary_parent: Dict[int, int] = {}
    for k, tet in enumerate(np.asarray(original.tets, dtype=int)):
        bary = np.mean(original.coords[tet], axis=0)
        m = len(coords)
        coords.append(bary)
        bary_parent[m] = k
        a, b, c, d = map(int, tet)
        candidates = [[b, c, d, m], [a, d, c, m], [a, b, d, m], [a, c, b, m]]
        candidates = oriented_tets(np.asarray(coords), np.asarray(candidates, dtype=int))
        subtets.extend(candidates.tolist())
        parent.extend([k] * 4)
    return TetComplex(coords=np.asarray(coords, dtype=float), tets=np.asarray(subtets, dtype=int), parent=np.asarray(parent, dtype=int), n_original_vertices=original.coords.shape[0], bary_parent=bary_parent)

def extract_patch(complex_: TetComplex, global_element_ids: Sequence[int], parent_original_tet: Sequence[int] | None=None) -> Patch:
    gids = np.asarray(sorted(set(map(int, global_element_ids))), dtype=int)
    selected_tets_global_nodes = complex_.tets[gids]
    global_nodes = np.asarray(sorted(set(selected_tets_global_nodes.ravel().tolist())), dtype=int)
    g2l = {int(g): i for i, g in enumerate(global_nodes)}
    local_tets = np.vectorize(g2l.__getitem__, otypes=[int])(selected_tets_global_nodes)
    local_coords = complex_.coords[global_nodes]
    local_tets = oriented_tets(local_coords, local_tets)
    mesh = make_ngsolve_tet_mesh(local_coords, local_tets)
    if parent_original_tet is None:
        parents = gids.copy()
    else:
        parents = np.asarray(parent_original_tet, dtype=int)[gids]
    return Patch(mesh=mesh, coords=local_coords, tets=local_tets, local_to_global_nodes=global_nodes, global_to_local_nodes=g2l, local_to_global_elements=gids, parent_original_tet=parents)

def ngmat_to_scipy(mat) -> sp.csr_matrix:
    rows, cols, vals = mat.COO()
    return sp.csr_matrix((np.asarray(vals), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=(mat.height, mat.width))

def ngvec_numpy(vec) -> Array:
    return np.array(vec.FV().NumPy(), dtype=float, copy=True)

def set_ngvec(vec, values: Array) -> None:
    arr = vec.FV().NumPy()
    arr[:] = np.asarray(values, dtype=float)

def mesh_edge_dof_map(mesh: Mesh, fes, local_to_global_nodes: Array) -> Dict[Edge, int]:
    ans: Dict[Edge, int] = {}
    for edge in mesh.edges:
        local_vertices = [int(v.nr) for v in edge.vertices]
        global_vertices = [int(local_to_global_nodes[i]) for i in local_vertices]
        key = (min(global_vertices), max(global_vertices))
        dofs = tuple((int(d) for d in fes.GetDofNrs(NodeId(EDGE, int(edge.nr)))))
        if len(dofs) != 1:
            raise RuntimeError(f'Expected one lowest-order HCurl dof on edge {key}, got {dofs}. Use HCurl(order=0).')
        ans[key] = dofs[0]
    return ans

def mesh_face_dof_map(mesh: Mesh, fes, local_to_global_nodes: Array) -> Dict[Face, int]:
    ans: Dict[Face, int] = {}
    for face in mesh.faces:
        local_vertices = [int(v.nr) for v in face.vertices]
        global_vertices = [int(local_to_global_nodes[i]) for i in local_vertices]
        key = tuple(sorted(global_vertices))
        dofs = tuple((int(d) for d in fes.GetDofNrs(NodeId(FACE, int(face.nr)))))
        if len(dofs) != 1:
            raise RuntimeError(f'Expected one lowest-order HDiv dof on face {key}, got {dofs}. Use HDiv(order=0, RT=True).')
        ans[key] = dofs[0]
    return ans

def vertex_dof_map(mesh: Mesh, fes, local_to_global_nodes: Array) -> Dict[int, int]:
    ans: Dict[int, int] = {}
    for lv, gv in enumerate(local_to_global_nodes):
        dofs = tuple((int(d) for d in fes.GetDofNrs(NodeId(VERTEX, lv))))
        if len(dofs) != 1:
            raise RuntimeError('Expected one P1 dof per vertex')
        ans[int(gv)] = dofs[0]
    return ans

def make_mu_on_alfeld_patch(patch: Patch, n_original_vertices: int) -> GridFunction:
    fes = H1(patch.mesh, order=1)
    gf = GridFunction(fes, name='mu_e')
    gf.vec[:] = 0.0
    vmap = vertex_dof_map(patch.mesh, fes, patch.local_to_global_nodes)
    for global_node, dof in vmap.items():
        gf.vec[dof] = 0.0 if global_node < n_original_vertices else 1.0
    return gf

def h1_embedding_coarse_to_alfeld(coarse: Patch, fine: Patch, original: TetComplex, alfeld: TetComplex) -> Tuple[Array, object, object]:
    Vc = H1(coarse.mesh, order=1)
    Vf = H1(fine.mesh, order=1)
    cmap = vertex_dof_map(coarse.mesh, Vc, coarse.local_to_global_nodes)
    fmap = vertex_dof_map(fine.mesh, Vf, fine.local_to_global_nodes)
    E = np.zeros((Vf.ndof, Vc.ndof))
    for global_fine_node, fdof in fmap.items():
        if global_fine_node < original.coords.shape[0]:
            E[fdof, cmap[global_fine_node]] = 1.0
        else:
            if alfeld.bary_parent is None or global_fine_node not in alfeld.bary_parent:
                raise KeyError(f'Missing parent for Alfeld barycenter node {global_fine_node}')
            parent = alfeld.bary_parent[global_fine_node]
            for gv in original.tets[parent]:
                E[fdof, cmap[int(gv)]] += 0.25
    return (E, Vc, Vf)

def barycentric_endpoint(global_node: int, parent_tet: int, original: TetComplex, alfeld: TetComplex) -> Array:
    tet = list(map(int, original.tets[parent_tet]))
    lam = np.zeros(4)
    if global_node < original.coords.shape[0]:
        lam[tet.index(global_node)] = 1.0
    else:
        if alfeld.bary_parent is None or alfeld.bary_parent[global_node] != parent_tet:
            raise ValueError('Fine edge endpoint is not in the requested parent tetrahedron')
        lam[:] = 0.25
    return lam

def parent_for_fine_edge(edge: Edge, selected_parents: Sequence[int], original: TetComplex, alfeld: TetComplex) -> int:
    p, q = edge
    bary_nodes = [g for g in edge if g >= original.coords.shape[0]]
    if bary_nodes:
        if len(bary_nodes) != 1:
            raise ValueError('An Alfeld edge cannot connect two distinct barycenters')
        assert alfeld.bary_parent is not None
        parent = int(alfeld.bary_parent[bary_nodes[0]])
        if parent not in set(map(int, selected_parents)):
            raise ValueError('Barycenter parent is outside patch')
        return parent
    for parent in selected_parents:
        tetset = set(map(int, original.tets[int(parent)]))
        if p in tetset and q in tetset:
            return int(parent)
    raise ValueError(f'Could not find parent tetrahedron for fine edge {edge}')

def hcurl_embedding_coarse_to_alfeld(coarse: Patch, fine: Patch, original: TetComplex, alfeld: TetComplex) -> Tuple[Array, object, object, Dict[Edge, int], Dict[Edge, int]]:
    Wc = HCurl(coarse.mesh, order=0)
    Wf = HCurl(fine.mesh, order=0)
    cmap = mesh_edge_dof_map(coarse.mesh, Wc, coarse.local_to_global_nodes)
    fmap = mesh_edge_dof_map(fine.mesh, Wf, fine.local_to_global_nodes)
    E = np.zeros((Wf.ndof, Wc.ndof))
    selected_parents = sorted(set(map(int, coarse.local_to_global_elements)))
    for fine_edge, fdof in fmap.items():
        p, q = fine_edge
        parent = parent_for_fine_edge(fine_edge, selected_parents, original, alfeld)
        tet = list(map(int, original.tets[parent]))
        lp = barycentric_endpoint(p, parent, original, alfeld)
        lq = barycentric_endpoint(q, parent, original, alfeld)
        for ii, jj in combinations(range(4), 2):
            gi, gj = (tet[ii], tet[jj])
            if gi < gj:
                edge = (gi, gj)
                val = lp[ii] * lq[jj] - lp[jj] * lq[ii]
            else:
                edge = (gj, gi)
                val = -(lp[ii] * lq[jj] - lp[jj] * lq[ii])
            if edge in cmap:
                E[fdof, cmap[edge]] += val
    return (E, Wc, Wf, cmap, fmap)

def l2_cell_gridfunction(patch: Patch, values: Sequence[float], name: str) -> GridFunction:
    values = np.asarray(values, dtype=float)
    if values.size != patch.tets.shape[0]:
        raise ValueError('One P0 value is required per patch tetrahedron')
    Q = L2(patch.mesh, order=0)
    gf = GridFunction(Q, name=name)
    gf.vec[:] = 0.0
    for local_el, val in enumerate(values):
        dofs = tuple((int(d) for d in Q.GetDofNrs(ElementId(VOL, local_el))))
        if len(dofs) != 1:
            raise RuntimeError('Expected one L2(order=0) dof per tetrahedron')
        gf.vec[dofs[0]] = float(val)
    return gf

def build_z0_weight(vertex: int, original: TetComplex, alfeld: TetComplex, compute_diagnostics: bool=True) -> Z0Weight:
    star_tets = np.where(np.any(original.tets == int(vertex), axis=1))[0]
    fine_cells = np.where(np.isin(alfeld.parent, star_tets))[0]
    coarse = extract_patch(original, star_tets)
    fine = extract_patch(alfeld, fine_cells, alfeld.parent)
    mu = make_mu_on_alfeld_patch(fine, original.coords.shape[0])
    E0, Vc, Vf = h1_embedding_coarse_to_alfeld(coarse, fine, original, alfeld)
    uf, vf = Vf.TnT()
    astiff = BilinearForm(Vf)
    astiff += mu * grad(uf) * grad(vf) * dx
    astiff.Assemble()
    Af = ngmat_to_scipy(astiff.mat).toarray()
    A = E0.T @ Af @ E0
    oneform = LinearForm(Vf)
    oneform += vf * dx
    oneform.Assemble()
    mf = ngvec_numpy(oneform.vec)
    m = E0.T @ mf
    volume = float(Integrate(1.0, fine.mesh))
    coarse_vmap = vertex_dof_map(coarse.mesh, Vc, coarse.local_to_global_nodes)
    q = np.zeros(Vc.ndof)
    q[coarse_vmap[int(vertex)]] = 1.0
    rhs = q - m / volume
    KKT = np.block([[A, m[:, None]], [m[None, :], np.zeros((1, 1))]])
    sol = sla.solve(KKT, np.r_[rhs, 0.0], assume_a='sym')
    psi_c = sol[:-1]
    psi_f = E0 @ psi_c
    gfpsi = GridFunction(Vf, name=f'psi0_v{vertex}')
    set_ngvec(gfpsi.vec, psi_f)
    z0_cf = 1.0 / volume - grad(mu) * grad(gfpsi)
    Qf = L2(fine.mesh, order=0)
    gfz0 = GridFunction(Qf, name=f'Z0_v{vertex}')
    gfz0.Set(z0_cf)
    if compute_diagnostics:
        z0_moment = LinearForm(Vf)
        z0_moment += gfz0 * vf * dx
        z0_moment.Assemble()
        coarse_moments = E0.T @ ngvec_numpy(z0_moment.vec)
        reproduction_error = float(np.linalg.norm(coarse_moments - q, ord=np.inf))
        integral_error = abs(float(Integrate(gfz0, fine.mesh)) - 1.0)
        zero_mean_psi_error = abs(float(np.dot(m, psi_c)))
    else:
        reproduction_error = np.nan
        integral_error = np.nan
        zero_mean_psi_error = np.nan
    global_values = np.zeros(alfeld.tets.shape[0])
    local_coeffs = ngvec_numpy(gfz0.vec)
    for local_el, global_el in enumerate(fine.local_to_global_elements):
        dof = int(Qf.GetDofNrs(ElementId(VOL, local_el))[0])
        global_values[int(global_el)] = local_coeffs[dof]
    return Z0Weight(vertex=int(vertex), values_global_alfeld_cells=global_values, reproduction_error=reproduction_error, integral_error=integral_error, zero_mean_psi_error=zero_mean_psi_error)

def solve_eta1_mixed(fine: Patch, rhs_values: Array, compute_diagnostics: bool=True) -> Tuple[GridFunction, GridFunction, GridFunction, Dict[str, float]]:
    V = HDiv(fine.mesh, order=0, RT=True, dirichlet='patch_boundary')
    Q = L2(fine.mesh, order=0)
    R = NumberSpace(fine.mesh)
    X = V * Q * R
    eta, u, lam = X.TrialFunction()
    v, w, mu0 = X.TestFunction()
    a = BilinearForm(X)
    a += (eta * v + u * div(v) + div(eta) * w + lam * w + mu0 * u) * dx
    a.Assemble()
    gfr = l2_cell_gridfunction(fine, rhs_values, 'Z0_boundary')
    f = LinearForm(X)
    f += -gfr * w * dx
    f.Assemble()
    gfu = GridFunction(X, name='eta1_mixed')
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    try:
        inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse='umfpack')
    except Exception:
        inv = a.mat.Inverse(freedofs=X.FreeDofs(), inverse='sparsecholesky')
    gfu.vec.data += inv * res
    gfeta, gfp, gflam = gfu.components
    if compute_diagnostics:
        div_error = float(Integrate((div(gfeta) + gfr) ** 2, fine.mesh) ** 0.5)
        normal = specialcf.normal(3)
        bflux = float(Integrate((gfeta.Trace() * normal) ** 2, fine.mesh, BND) ** 0.5)
        zero_mean_u = abs(float(Integrate(gfp, fine.mesh)))
        lambda_value = abs(float(ngvec_numpy(gflam.vec)[0]))
        diagnostics = {'div_error': div_error, 'boundary_flux_norm': bflux, 'zero_mean_u': zero_mean_u, 'lambda_value': lambda_value}
    else:
        diagnostics = {}
    return (gfeta, gfp, gfr, diagnostics)

def solve_psi1_and_build_z1(edge: Edge, original: TetComplex, alfeld: TetComplex, z0_weights: Dict[int, Z0Weight], compute_diagnostics: bool=True) -> Z1Result:
    a, b = map(int, edge)
    if a > b:
        a, b = (b, a)
    edge = (a, b)
    original_edges = set(canonical_edges(original.tets))
    if edge not in original_edges:
        raise ValueError(f'{edge} is not an original mesh edge')
    patch_tets = np.where(np.any(original.tets == a, axis=1) | np.any(original.tets == b, axis=1))[0]
    patch_cells = np.where(np.isin(alfeld.parent, patch_tets))[0]
    coarse = extract_patch(original, patch_tets)
    fine = extract_patch(alfeld, patch_cells, alfeld.parent)
    mu = make_mu_on_alfeld_patch(fine, original.coords.shape[0])
    z0a = z0_weights[a].values_global_alfeld_cells[patch_cells]
    z0b = z0_weights[b].values_global_alfeld_cells[patch_cells]
    rhs_values = z0b - z0a
    eta, multiplier_u, gfr, eta_diag = solve_eta1_mixed(fine, rhs_values, compute_diagnostics=compute_diagnostics)
    E1, Wc, Wf, coarse_edge_dofs, _ = hcurl_embedding_coarse_to_alfeld(coarse, fine, original, alfeld)
    uf, vf = Wf.TnT()
    af = BilinearForm(Wf)
    af += mu * curl(uf) * curl(vf) * dx
    af.Assemble()
    Af = ngmat_to_scipy(af.mat).toarray()
    mf = BilinearForm(Wf)
    mf += uf * vf * dx
    mf.Assemble()
    Mf = ngmat_to_scipy(mf.mat).toarray()
    leta = LinearForm(Wf)
    leta += eta * vf * dx
    leta.Assemble()
    eta_fine_moments = ngvec_numpy(leta.vec)
    A = E1.T @ Af @ E1
    M = E1.T @ Mf @ E1
    eta_coarse_moments = E1.T @ eta_fine_moments
    phi = np.zeros(Wc.ndof)
    phi[coarse_edge_dofs[edge]] = 1.0
    rhs = phi - eta_coarse_moments
    As = 0.5 * (A + A.T)
    evals, evecs = sla.eigh(As)
    tol = max(1.0, float(np.max(np.abs(evals)))) * 1e-11
    K = evecs[:, evals <= tol]
    compat = float(np.linalg.norm(K.T @ rhs, ord=np.inf)) if (compute_diagnostics and K.shape[1]) else (0.0 if compute_diagnostics else np.nan)
    MK = M @ K
    KKT = np.block([[A, MK], [K.T @ M, np.zeros((K.shape[1], K.shape[1]))]])
    rhs_kkt = np.r_[rhs, np.zeros(K.shape[1])]
    try:
        sol = sla.solve(KKT, rhs_kkt, assume_a='sym')
    except sla.LinAlgError:
        sol = sla.lstsq(KKT, rhs_kkt, cond=1e-12)[0]
    psi_c = sol[:Wc.ndof]
    psi_f = E1 @ psi_c
    gfpsi = GridFunction(Wf, name='psi1')
    set_ngvec(gfpsi.vec, psi_f)
    correction = Cross(grad(mu), curl(gfpsi))
    z1_cf = eta + correction
    Vall = HDiv(fine.mesh, order=0, RT=True)
    gfz1 = GridFunction(Vall, name='Z1')
    gfz1.Set(z1_cf)

    if compute_diagnostics:
        psi_eq_res = float(np.linalg.norm(A @ psi_c - rhs, ord=np.inf))
        psi_orth = float(np.linalg.norm(K.T @ M @ psi_c, ord=np.inf)) if K.shape[1] else 0.0
        z1_div_error = float(Integrate((div(gfz1) + gfr) ** 2, fine.mesh) ** 0.5)
        normal = specialcf.normal(3)
        z1_bflux = float(Integrate((gfz1.Trace() * normal) ** 2, fine.mesh, BND) ** 0.5)
        reproduction = eta_coarse_moments + A @ psi_c
        reproduction_error = float(np.linalg.norm(reproduction - phi, ord=np.inf))
        eta_norm = float(Integrate(eta * eta, fine.mesh) ** 0.5)
        rhs_norm = float(Integrate(gfr * gfr, fine.mesh) ** 0.5)
        h_e = float(np.linalg.norm(original.coords[b] - original.coords[a]))
        stability_ratio = eta_norm / (h_e * rhs_norm) if rhs_norm > 0 else 0.0
        eta_div_error = eta_diag['div_error']
        eta_boundary_flux_norm = eta_diag['boundary_flux_norm']
        eta_mean_multiplier = eta_diag['zero_mean_u']
    else:
        psi_eq_res = np.nan
        psi_orth = np.nan
        z1_div_error = np.nan
        z1_bflux = np.nan
        reproduction_error = np.nan
        stability_ratio = np.nan
        eta_div_error = np.nan
        eta_boundary_flux_norm = np.nan
        eta_mean_multiplier = np.nan

    return Z1Result(edge=edge, original_patch_tets=patch_tets, alfeld_patch_cells=patch_cells, fine_patch=fine, coarse_patch=coarse, mu=mu, rhs_z0_boundary=gfr, eta1=eta, multiplier_u=multiplier_u, psi1_fine=gfpsi, z1=gfz1, eta_div_error=eta_div_error, eta_boundary_flux_norm=eta_boundary_flux_norm, eta_mean_multiplier=eta_mean_multiplier, eta_stability_ratio=stability_ratio, psi_equation_error=psi_eq_res, psi_kernel_compatibility=compat, psi_kernel_orthogonality=psi_orth, z1_div_error=z1_div_error, z1_boundary_flux_norm=z1_bflux, reproduction_error=reproduction_error)

@dataclass
class Pi1MatrixResult:
    original: TetComplex
    alfeld: TetComplex
    coarse_patch: Patch
    fine_patch: Patch
    coarse_h1: object
    fine_h1: object
    coarse_hcurl: object
    fine_hcurl: object
    fine_hdiv: object
    coarse_edge_dofs: Dict[Edge, int]
    fine_edge_dofs: Dict[Edge, int]
    fine_face_dofs: Dict[Face, int]
    global_z1_weights: Dict[Edge, GridFunction]
    P0: Array
    P1: Array
    E0: Array
    E1: Array
    G0: Array
    GA: Array
    z0_weights: Dict[int, Z0Weight]
    z1_results: Dict[Edge, Z1Result]
    diagnostics: Dict[str, float]

def full_patch(complex_: TetComplex) -> Patch:
    ids = np.arange(complex_.tets.shape[0], dtype=int)
    parents = complex_.parent if complex_.parent is not None else ids
    return extract_patch(complex_, ids, parents)

def discrete_gradient_matrix(mesh: Mesh, V0, V1) -> Array:
    G = np.zeros((V1.ndof, V0.ndof))
    g0 = GridFunction(V0)
    g1 = GridFunction(V1)
    for j in range(V0.ndof):
        g0.vec[:] = 0.0
        g0.vec[j] = 1.0
        g1.vec[:] = 0.0
        g1.Set(grad(g0))
        G[:, j] = ngvec_numpy(g1.vec)
    return G

def assemble_global_p0(original: TetComplex, alfeld: TetComplex, coarse: Patch, fine: Patch, z0_weights: Dict[int, Z0Weight]) -> Tuple[Array, Array, object, object]:
    E0, Vc, Vf = h1_embedding_coarse_to_alfeld(coarse, fine, original, alfeld)
    coarse_vmap = vertex_dof_map(coarse.mesh, Vc, coarse.local_to_global_nodes)
    P0 = np.zeros((Vc.ndof, Vf.ndof))
    _, vf = Vf.TnT()
    for vertex, weight in z0_weights.items():
        z0_global = l2_cell_gridfunction(fine, weight.values_global_alfeld_cells, name=f'Z0_global_v{vertex}')
        lf = LinearForm(Vf)
        lf += z0_global * vf * dx
        lf.Assemble()
        P0[coarse_vmap[int(vertex)], :] = ngvec_numpy(lf.vec)
    return (P0, E0, Vc, Vf)

def assemble_global_p1_row(z1_result: Z1Result, global_fine_hcurl, global_fine_edge_dofs: Dict[Edge, int]) -> Array:
    Wpatch = z1_result.psi1_fine.space
    _, vpatch = Wpatch.TnT()
    lf = LinearForm(Wpatch)
    lf += z1_result.z1 * vpatch * dx
    lf.Assemble()
    local_moments = ngvec_numpy(lf.vec)
    local_edge_dofs = mesh_edge_dof_map(z1_result.fine_patch.mesh, Wpatch, z1_result.fine_patch.local_to_global_nodes)
    row = np.zeros(global_fine_hcurl.ndof)
    for fine_edge, local_dof in local_edge_dofs.items():
        if fine_edge not in global_fine_edge_dofs:
            raise KeyError(f'Patch Alfeld edge {fine_edge} is absent from global Alfeld mesh')
        row[global_fine_edge_dofs[fine_edge]] += local_moments[local_dof]
    return row

def scatter_local_z1_to_global_alfeld(z1_result: Z1Result, global_fine_hdiv, global_fine_face_dofs: Dict[Face, int], name: str) -> GridFunction:
    local_space = z1_result.z1.space
    local_face_dofs = mesh_face_dof_map(z1_result.fine_patch.mesh, local_space, z1_result.fine_patch.local_to_global_nodes)
    global_weight = GridFunction(global_fine_hdiv, name=name)
    global_weight.vec[:] = 0.0
    local_coefficients = ngvec_numpy(z1_result.z1.vec)
    for face, local_dof in local_face_dofs.items():
        if face not in global_fine_face_dofs:
            raise KeyError(f'Patch Alfeld face {face} is absent from the global Alfeld mesh')
        global_dof = global_fine_face_dofs[face]
        global_weight.vec[global_dof] = local_coefficients[local_dof]
    return global_weight

def assemble_global_weight_hcurl_row(global_weight: GridFunction, global_hcurl) -> Array:
    _, test = global_hcurl.TnT()
    linear_form = LinearForm(global_hcurl)
    linear_form += InnerProduct(global_weight, test) * dx
    linear_form.Assemble()
    return ngvec_numpy(linear_form.vec)

def build_faithful_pi1_matrix(original: TetComplex, tolerance: float=5e-09, retain_local_weights: bool=True, verbose: bool=True) -> Pi1MatrixResult:
    alfeld = alfeld_split(original)
    coarse = full_patch(original)
    fine = full_patch(alfeld)
    vertices = list(range(original.coords.shape[0]))
    edges = canonical_edges(original.tets)
    if verbose:
        print(f'Building {len(vertices)} faithful Z0 vertex weights ...')
    z0_weights: Dict[int, Z0Weight] = {}
    for vertex in vertices:
        z0_weights[vertex] = build_z0_weight(vertex, original, alfeld)
        if verbose:
            z = z0_weights[vertex]
            print(f'  Z0({vertex}): repro={z.reproduction_error:.3e}, integral={z.integral_error:.3e}, mean(psi0)={z.zero_mean_psi_error:.3e}')
    P0, E0, V0c, V0A = assemble_global_p0(original, alfeld, coarse, fine, z0_weights)
    E1, V1c, V1A, coarse_edge_dofs, fine_edge_dofs = hcurl_embedding_coarse_to_alfeld(coarse, fine, original, alfeld)
    V2A = HDiv(fine.mesh, order=0, RT=True)
    fine_face_dofs = mesh_face_dof_map(fine.mesh, V2A, fine.local_to_global_nodes)
    G0 = discrete_gradient_matrix(coarse.mesh, V0c, V1c)
    GA = discrete_gradient_matrix(fine.mesh, V0A, V1A)
    P1 = np.zeros((V1c.ndof, V1A.ndof))
    z1_results: Dict[Edge, Z1Result] = {}
    global_z1_weights: Dict[Edge, GridFunction] = {}
    max_local = 0.0
    max_global_scatter_row_error = 0.0
    if verbose:
        print(f'\nBuilding {len(edges)} faithful eta1/psi1/Z1 edge weights ...')
    for edge in edges:
        result = solve_psi1_and_build_z1(edge, original, alfeld, z0_weights)
        local_row = assemble_global_p1_row(result, V1A, fine_edge_dofs)
        global_weight = scatter_local_z1_to_global_alfeld(result, V2A, fine_face_dofs, name=f'Z1_global_e{edge[0]}_{edge[1]}')
        global_row = assemble_global_weight_hcurl_row(global_weight, V1A)
        scatter_row_error = float(np.linalg.norm(global_row - local_row, ord=np.inf))
        max_global_scatter_row_error = max(max_global_scatter_row_error, scatter_row_error)
        P1[coarse_edge_dofs[edge], :] = global_row
        global_z1_weights[edge] = global_weight
        local_checks = [result.eta_div_error, result.eta_boundary_flux_norm, result.eta_mean_multiplier, result.psi_kernel_compatibility, result.psi_kernel_orthogonality, result.psi_equation_error, result.z1_div_error, result.z1_boundary_flux_norm, result.reproduction_error]
        local_worst = float(max(local_checks))
        max_local = max(max_local, local_worst)
        if verbose:
            print(f'  edge {edge}: eta-div={result.eta_div_error:.2e}, psi={result.psi_equation_error:.2e}, Z1-div={result.z1_div_error:.2e}, repro={result.reproduction_error:.2e}')
        if retain_local_weights:
            z1_results[edge] = result
    I0 = np.eye(V0c.ndof)
    I1 = np.eye(V1c.ndof)
    p0e0 = P0 @ E0 - I0
    p1e1 = P1 @ E1 - I1
    commute = P1 @ GA - G0 @ P0
    Q1 = E1 @ P1
    idempotence = Q1 @ Q1 - Q1
    rng = np.random.default_rng(20260717)
    c1 = rng.standard_normal(V1c.ndof)
    c0 = rng.standard_normal(V0A.ndof)
    random_reproduction = float(np.linalg.norm(P1 @ (E1 @ c1) - c1, ord=np.inf))
    random_commutation = float(np.linalg.norm(P1 @ (GA @ c0) - G0 @ (P0 @ c0), ord=np.inf))
    diagnostics = {'max_local_weight_residual': max_local, 'global_z1_scatter_row_inf': max_global_scatter_row_error, 'P0E0_max_entry': float(np.max(np.abs(p0e0))), 'P0E0_inf': float(np.linalg.norm(p0e0, ord=np.inf)), 'P1E1_max_entry': float(np.max(np.abs(p1e1))), 'P1E1_inf': float(np.linalg.norm(p1e1, ord=np.inf)), 'commute_max_entry': float(np.max(np.abs(commute))), 'commute_inf': float(np.linalg.norm(commute, ord=np.inf)), 'fine_projection_idempotence_inf': float(np.linalg.norm(idempotence, ord=np.inf)), 'random_original_hcurl_reproduction': random_reproduction, 'random_gradient_commutation': random_commutation}
    if verbose:
        print('\nGlobal Pi1/P1 checks')
        print(f'  original H1 / HCurl dofs    : {V0c.ndof} / {V1c.ndof}')
        print(f'  Alfeld H1 / HCurl dofs      : {V0A.ndof} / {V1A.ndof}')
        print(f'  P1 shape                    : {P1.shape}')
        print(f'  max local weight residual   : {max_local:.3e}')
        print(f'  global Z1 scatter row error : {max_global_scatter_row_error:.3e}')
        print(f"  ||P0 E0-I||_inf             : {diagnostics['P0E0_inf']:.3e}")
        print(f"  ||P1 E1-I||_inf             : {diagnostics['P1E1_inf']:.3e}")
        print(f"  ||P1 GA-G0 P0||_inf         : {diagnostics['commute_inf']:.3e}")
        print(f"  ||(E1P1)^2-E1P1||_inf       : {diagnostics['fine_projection_idempotence_inf']:.3e}")
        print(f'  random HCurl reproduction   : {random_reproduction:.3e}')
        print(f'  random grad commutation     : {random_commutation:.3e}')
    worst_global = max(max_local, max_global_scatter_row_error, diagnostics['P0E0_inf'], diagnostics['P1E1_inf'], diagnostics['commute_inf'], diagnostics['fine_projection_idempotence_inf'], random_reproduction, random_commutation)
    if worst_global > tolerance:
        raise RuntimeError(f'Faithful Pi1 construction failed a check: worst={worst_global:.3e}, tolerance={tolerance:.3e}')
    return Pi1MatrixResult(original=original, alfeld=alfeld, coarse_patch=coarse, fine_patch=fine, coarse_h1=V0c, fine_h1=V0A, coarse_hcurl=V1c, fine_hcurl=V1A, fine_hdiv=V2A, coarse_edge_dofs=coarse_edge_dofs, fine_edge_dofs=fine_edge_dofs, fine_face_dofs=fine_face_dofs, global_z1_weights=global_z1_weights, P0=P0, P1=P1, E0=E0, E1=E1, G0=G0, GA=GA, z0_weights=z0_weights, z1_results=z1_results, diagnostics=diagnostics)

def apply_pi1_to_l2_field(result: Pi1MatrixResult, u, name: str='Pi1_L2_u', integration_order: int | None=None) -> Tuple[GridFunction, Array]:
    dim = getattr(u, 'dim', None)
    if dim != 3:
        raise ValueError(f'Pi_0^1 expects a three-component vector field; received dim={dim!r}.')
    if not result.global_z1_weights:
        raise RuntimeError('Global Z1 weights were not retained, so direct L2 application is unavailable.')
    coefficients = np.zeros(result.coarse_hcurl.ndof)
    for edge, weight in result.global_z1_weights.items():
        integrand = InnerProduct(weight, u)
        try:
            if integration_order is None:
                value = Integrate(integrand, result.fine_patch.mesh)
            else:
                value = Integrate(integrand, result.fine_patch.mesh, order=int(integration_order))
        except Exception as exc:
            raise ValueError('The supplied field could not be evaluated on the global Alfeld integration mesh. Pass an analytic CoefficientFunction or a GridFunction whose space is built on result.fine_patch.mesh.') from exc
        coefficients[result.coarse_edge_dofs[edge]] = float(value)
    projected = gridfunction_from_native_coefficients(result.coarse_hcurl, coefficients, name=name)
    storage_error = float(np.linalg.norm(ngvec_numpy(projected.vec) - coefficients, ord=np.inf))
    if storage_error > 100.0 * np.finfo(float).eps:
        raise RuntimeError(f'The coarse HCurl GridFunction did not retain the directly computed Pi_0^1 coefficients: error={storage_error:.3e}.')
    return (projected, coefficients)

def gridfunction_from_native_coefficients(fes, coefficients: Array, name: str) -> GridFunction:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (fes.ndof,):
        raise ValueError(f'Expected a coefficient vector of shape ({fes.ndof},), got {coefficients.shape}.')
    gf = GridFunction(fes, name=name)
    set_ngvec(gf.vec, coefficients)
    return gf

def verify_pi1_direct_l2_application(result: Pi1MatrixResult, seed: int=20260722, tolerance: float=5e-09, verbose: bool=True) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    fine_coefficients = rng.standard_normal(result.fine_hcurl.ndof)
    u_alfeld = gridfunction_from_native_coefficients(result.fine_hcurl, fine_coefficients, name='u_alfeld_direct_test')
    _, direct_coefficients = apply_pi1_to_l2_field(result, u_alfeld, name='Pi1_direct_u_alfeld')
    matrix_coefficients = result.P1 @ fine_coefficients
    direct_vs_matrix = float(np.linalg.norm(direct_coefficients - matrix_coefficients, ord=np.inf))
    coarse_coefficients = rng.standard_normal(result.coarse_hcurl.ndof)
    embedded_coefficients = result.E1 @ coarse_coefficients
    embedded = gridfunction_from_native_coefficients(result.fine_hcurl, embedded_coefficients, name='E1_u_coarse_direct_test')
    projected, reproduced_coefficients = apply_pi1_to_l2_field(result, embedded, name='Pi1_direct_E1_u_coarse')
    coarse_reproduction = float(np.linalg.norm(reproduced_coefficients - coarse_coefficients, ord=np.inf))
    storage_error = float(np.linalg.norm(ngvec_numpy(projected.vec) - reproduced_coefficients, ord=np.inf))
    vector_l2 = L2(result.fine_patch.mesh, order=1, dim=3)
    u_l2 = GridFunction(vector_l2, name='arbitrary_vector_L2_input')
    set_ngvec(u_l2.vec, rng.standard_normal(u_l2.vec.FV().NumPy().size))
    _, l2_coefficients = apply_pi1_to_l2_field(result, u_l2, name='Pi1_arbitrary_vector_L2')
    l2_output_finite = float(0.0 if np.all(np.isfinite(l2_coefficients)) else np.inf)
    analytic_u = CoefficientFunction((1.0 + x + y, 2.0 - y + z, x - 2.0 * z))
    _, analytic_coefficients = apply_pi1_to_l2_field(result, analytic_u, name='Pi1_analytic_L2_field')
    analytic_output_finite = float(0.0 if np.all(np.isfinite(analytic_coefficients)) else np.inf)
    diagnostics = {'direct_vs_P1_on_alfeld_hcurl_inf': direct_vs_matrix, 'direct_coarse_reproduction_inf': coarse_reproduction, 'direct_gridfunction_storage_inf': storage_error, 'arbitrary_vector_L2_output_finite': l2_output_finite, 'analytic_coefficientfunction_output_finite': analytic_output_finite}
    if verbose:
        print('\nDirect [L2(Omega)]^3 Pi_0^1 application checks')
        print(f'  direct weights vs P1 on Alfeld HCurl : {direct_vs_matrix:.3e}')
        print(f'  direct Pi1(E1 u_h) reproduction       : {coarse_reproduction:.3e}')
        print(f'  direct coarse GridFunction storage    : {storage_error:.3e}')
        print(f'  discontinuous vector L2 input accepted : {l2_output_finite == 0.0}')
        print(f'  analytic CoefficientFunction accepted  : {analytic_output_finite == 0.0}')
    worst = max(diagnostics.values())
    if worst > tolerance:
        raise RuntimeError(f'Direct [L2]^3 Pi_0^1 application failed: worst={worst:.3e}, tolerance={tolerance:.3e}.')
    return diagnostics


@dataclass
class Pi1ApplicationResult:
    """Only the data required to apply Pi_0^1; no verification matrices."""
    original: TetComplex
    alfeld: TetComplex
    coarse_patch: Patch
    fine_patch: Patch
    coarse_hcurl: object
    coarse_edge_dofs: Dict[Edge, int]
    global_z1_weights: Dict[Edge, GridFunction]


def build_pi1_application_operator(original: TetComplex) -> Pi1ApplicationResult:
    """Build only the data needed to apply Pi_0^1.

    This path intentionally does not assemble convergence tests, commuting
    matrices, reproduction residuals, idempotence checks, or error diagnostics.
    """
    alfeld = alfeld_split(original)
    coarse = full_patch(original)
    fine = full_patch(alfeld)

    z0_weights: Dict[int, Z0Weight] = {}
    for vertex in range(original.coords.shape[0]):
        z0_weights[vertex] = build_z0_weight(
            vertex, original, alfeld, compute_diagnostics=False
        )

    coarse_hcurl = HCurl(coarse.mesh, order=0)
    coarse_edge_dofs = mesh_edge_dof_map(
        coarse.mesh, coarse_hcurl, coarse.local_to_global_nodes
    )

    fine_hdiv = HDiv(fine.mesh, order=0, RT=True)
    fine_face_dofs = mesh_face_dof_map(
        fine.mesh, fine_hdiv, fine.local_to_global_nodes
    )

    global_z1_weights: Dict[Edge, GridFunction] = {}
    for edge in canonical_edges(original.tets):
        z1_result = solve_psi1_and_build_z1(
            edge,
            original,
            alfeld,
            z0_weights,
            compute_diagnostics=False,
        )
        global_z1_weights[edge] = scatter_local_z1_to_global_alfeld(
            z1_result,
            fine_hdiv,
            fine_face_dofs,
            name=f'Z1_global_e{edge[0]}_{edge[1]}',
        )

    return Pi1ApplicationResult(
        original=original,
        alfeld=alfeld,
        coarse_patch=coarse,
        fine_patch=fine,
        coarse_hcurl=coarse_hcurl,
        coarse_edge_dofs=coarse_edge_dofs,
        global_z1_weights=global_z1_weights,
    )


def apply_pi1_application(
    result: Pi1ApplicationResult,
    u,
    name: str='Pi1_u_internal',
    integration_order: int | None=8,
) -> Tuple[GridFunction, Array]:
    """Apply Pi_0^1 without running any verification/error calculations."""
    dim = getattr(u, 'dim', None)
    if dim != 3:
        raise ValueError(
            f'Pi_0^1 expects a three-component vector field; received dim={dim!r}.'
        )

    coefficients = np.zeros(result.coarse_hcurl.ndof)
    for edge, weight in result.global_z1_weights.items():
        integrand = InnerProduct(weight, u)
        if integration_order is None:
            value = Integrate(integrand, result.fine_patch.mesh)
        else:
            value = Integrate(
                integrand,
                result.fine_patch.mesh,
                order=int(integration_order),
            )
        coefficients[result.coarse_edge_dofs[edge]] = float(value)

    projected = gridfunction_from_native_coefficients(
        result.coarse_hcurl, coefficients, name=name
    )
    return projected, coefficients


# ============================================================================
# Public NGSolve-mesh interface
# ============================================================================

def _tetcomplex_from_ngsolve_mesh(mesh: Mesh) -> TetComplex:
    """Convert a straight-sided tetrahedral NGSolve Mesh to the internal complex.

    Vertex numbering is preserved so that lowest-order HCurl edge orientations on
    the internal mesh agree with those on the user's original NGSolve mesh.
    """
    vertices = list(mesh.vertices)
    if not vertices:
        raise ValueError("The supplied NGSolve mesh has no vertices.")

    nvert = max(int(v.nr) for v in vertices) + 1
    coords = np.empty((nvert, 3), dtype=float)
    seen = np.zeros(nvert, dtype=bool)

    for v in vertices:
        p = mesh[v].point
        if len(p) != 3:
            raise ValueError("Pi_0^1 currently requires a three-dimensional mesh.")
        i = int(v.nr)
        coords[i, :] = np.asarray(p, dtype=float)
        seen[i] = True

    if not np.all(seen):
        raise ValueError(
            "The NGSolve mesh vertex numbering is not contiguous from 0 to nvert-1; "
            "this interface currently requires contiguous vertex numbering."
        )

    tets = []
    for el in mesh.Elements(VOL):
        verts = [int(v.nr) for v in el.vertices]
        if len(verts) != 4:
            raise ValueError(
                "Pi_0^1 currently requires a purely tetrahedral volume mesh; "
                f"element {int(el.nr)} has {len(verts)} vertices."
            )
        tets.append(verts)

    if not tets:
        raise ValueError("The supplied NGSolve mesh has no volume tetrahedra.")

    tets = oriented_tets(coords, np.asarray(tets, dtype=int))
    return TetComplex(
        coords=coords,
        tets=tets,
        n_original_vertices=nvert,
    )


class Pi1Projector:
    """Reusable lowest-order projection Pi_0^1 on an NGSolve tetrahedral mesh.

    The expensive Z0/Z1 construction is performed once when the object is
    created. Normal application mode builds only the operator data needed for
    Pi_0^1; it does not run convergence studies or verification/error tests.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self._mesh_data = _tetcomplex_from_ngsolve_mesh(mesh)
        self._result = build_pi1_application_operator(self._mesh_data)

        # Public target space on the user's original mesh.
        self.space = HCurl(mesh, order=0)

        identity_vertex_map = np.arange(self._mesh_data.coords.shape[0], dtype=int)
        self._user_edge_dofs = mesh_edge_dof_map(
            mesh,
            self.space,
            identity_vertex_map,
        )

        # Internal-to-user HCurl coefficient permutation.
        self._internal_to_user = np.empty(self._result.coarse_hcurl.ndof, dtype=int)
        self._internal_to_user.fill(-1)
        for edge, internal_dof in self._result.coarse_edge_dofs.items():
            if edge not in self._user_edge_dofs:
                raise RuntimeError(
                    f"Internal edge {edge} was not found on the supplied NGSolve mesh."
                )
            self._internal_to_user[int(internal_dof)] = int(self._user_edge_dofs[edge])

        if np.any(self._internal_to_user < 0):
            missing = np.where(self._internal_to_user < 0)[0]
            raise RuntimeError(
                "Could not map every internal HCurl degree of freedom back to the "
                f"user mesh; missing internal dofs {missing.tolist()}."
            )

    @property
    def ndof(self) -> int:
        return int(self.space.ndof)

    def project(
        self,
        u,
        name: str = "Pi1_u",
        integration_order: int | None = 8,
        return_coefficients: bool = False,
    ):
        """Return Pi_0^1 u as a GridFunction on the original user mesh."""
        _, internal_coeff = apply_pi1_application(
            self._result,
            u,
            name=name + "_internal",
            integration_order=integration_order,
        )

        coeff_user = np.zeros(self.space.ndof, dtype=float)
        for internal_dof, user_dof in enumerate(self._internal_to_user):
            coeff_user[int(user_dof)] = internal_coeff[int(internal_dof)]

        projected = gridfunction_from_native_coefficients(
            self.space,
            coeff_user,
            name=name,
        )

        if return_coefficients:
            return projected, coeff_user
        return projected

    def __call__(
        self,
        u,
        name: str = "Pi1_u",
        integration_order: int | None = 8,
        return_coefficients: bool = False,
    ):
        return self.project(
            u,
            name=name,
            integration_order=integration_order,
            return_coefficients=return_coefficients,
        )


def project_pi1(
    mesh: Mesh,
    u,
    name: str = "Pi1_u",
    integration_order: int | None = 8,
):
    """One-shot convenience wrapper for Pi_0^1."""
    projector = Pi1Projector(mesh)
    return projector(
        u,
        name=name,
        integration_order=integration_order,
    )


# ============================================================================
# Validation helpers
# ============================================================================
# These helpers support the separate validation notebook.  They are intentionally
# not part of the small public application API exported through __all__.

def structured_unit_cube_six_tets(n: int) -> TetComplex:
    """Uniform n x n x n unit-cube grid, each cube split into six tetrahedra.

    This deterministic mesh family is used only by the validation notebook to
    obtain a reproducible h-sequence.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")

    def node(i: int, j: int, k: int) -> int:
        return i * (n + 1) * (n + 1) + j * (n + 1) + k

    coords = np.array(
        [
            (i / n, j / n, k / n)
            for i in range(n + 1)
            for j in range(n + 1)
            for k in range(n + 1)
        ],
        dtype=float,
    )

    local_six = np.array(
        [
            [0, 4, 5, 7],
            [0, 5, 1, 7],
            [0, 1, 3, 7],
            [0, 3, 2, 7],
            [0, 2, 6, 7],
            [0, 6, 4, 7],
        ],
        dtype=int,
    )

    tets: List[List[int]] = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cube_nodes = np.array(
                    [
                        node(i, j, k),
                        node(i, j, k + 1),
                        node(i, j + 1, k),
                        node(i, j + 1, k + 1),
                        node(i + 1, j, k),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k),
                        node(i + 1, j + 1, k + 1),
                    ],
                    dtype=int,
                )
                tets.extend(cube_nodes[local_six].tolist())

    tets_array = oriented_tets(coords, np.asarray(tets, dtype=int))
    return TetComplex(
        coords=coords,
        tets=tets_array,
        n_original_vertices=coords.shape[0],
    )


def tetrahedral_hmax(mesh_data: TetComplex) -> float:
    """Largest edge length of a tetrahedral complex."""
    h = 0.0
    for tet in np.asarray(mesh_data.tets, dtype=int):
        pts = mesh_data.coords[tet]
        for a, b in combinations(range(4), 2):
            h = max(h, float(np.linalg.norm(pts[a] - pts[b])))
    return h


def ngmesh_from_data(mesh_data: TetComplex, boundary_name: str = "boundary") -> Mesh:
    """Create an NGSolve mesh from a TetComplex used by validation tests."""
    return make_ngsolve_tet_mesh(
        mesh_data.coords,
        mesh_data.tets,
        boundary_name=boundary_name,
    )


def apply_pi0_for_validation(
    result: Pi1MatrixResult,
    u,
    name: str = "Pi0_u",
    integration_order: int | None = 8,
) -> Tuple[GridFunction, Array]:
    """Apply the scalar Pi_0^0 associated with a full validation build.

    This is kept as a validation helper rather than part of the public
    application API.  It is used to test grad Pi_0^0 u = Pi_0^1 grad u.
    """
    coarse_vertex_dofs = vertex_dof_map(
        result.coarse_patch.mesh,
        result.coarse_h1,
        result.coarse_patch.local_to_global_nodes,
    )

    coefficients = np.zeros(result.coarse_h1.ndof)
    for vertex, weight in result.z0_weights.items():
        z0_global = l2_cell_gridfunction(
            result.fine_patch,
            weight.values_global_alfeld_cells,
            name=f"Z0_global_v{vertex}",
        )
        integrand = z0_global * u
        if integration_order is None:
            value = Integrate(integrand, result.fine_patch.mesh)
        else:
            value = Integrate(
                integrand,
                result.fine_patch.mesh,
                order=int(integration_order),
            )
        coefficients[coarse_vertex_dofs[int(vertex)]] = float(value)

    projected = gridfunction_from_native_coefficients(
        result.coarse_h1,
        coefficients,
        name=name,
    )
    return projected, coefficients


__all__ = ["Pi1Projector", "project_pi1"]
__version__ = "0.3.0"
