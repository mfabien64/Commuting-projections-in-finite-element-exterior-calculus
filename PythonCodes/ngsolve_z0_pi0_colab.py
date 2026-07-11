# Z^0 and Pi_h^0 construction in NGSolve/Netgen + NumPy/SciPy
# Colab cell 1, if needed:
#   !pip -q install ngsolve scipy matplotlib

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from ngsolve import Mesh, H1, GridFunction
from netgen.geom2d import unit_square


def extract_ngsolve_tri_mesh(mesh):
    """Return coords (nv,2), elems (ne,3) from an NGSolve 2D triangle mesh."""
    coords = np.array([[v.point[0], v.point[1]] for v in mesh.vertices], dtype=float)
    elems = []
    for el in mesh.Elements():
        verts = [v.nr for v in el.vertices]
        if len(verts) != 3:
            raise RuntimeError("This demo expects a triangular 2D mesh.")
        elems.append(verts)
    elems = np.array(elems, dtype=int)
    elems = orient_tris_ccw(coords, elems)
    return coords, elems


def orient_tris_ccw(coords, elems):
    elems = elems.copy()
    for k in range(elems.shape[0]):
        X = coords[elems[k], :]
        sarea2 = np.linalg.det(np.vstack([X[1] - X[0], X[2] - X[0]]))
        if sarea2 < 0:
            elems[k, [1, 2]] = elems[k, [2, 1]]
    return elems


def tri_area_grads(X):
    """Triangle area and gradients of P1 barycentric basis."""
    x1, y1 = X[0]
    x2, y2 = X[1]
    x3, y3 = X[2]
    twoA = np.linalg.det(np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]]))
    area = 0.5 * twoA
    if area <= 0:
        raise RuntimeError("nonpositive triangle area")
    grads = np.zeros((3, 2))
    grads[0] = np.array([y2 - y3, x3 - x2]) / twoA
    grads[1] = np.array([y3 - y1, x1 - x3]) / twoA
    grads[2] = np.array([y1 - y2, x2 - x1]) / twoA
    return area, grads


def barycentric_coords(P, X):
    """Barycentric coordinates of points P with respect to triangle X."""
    M = np.vstack([X.T, np.ones(3)])
    R = np.vstack([P.T, np.ones(P.shape[0])])
    return np.linalg.solve(M, R).T


def alfeld_split_2d(coords, elems):
    """Alfeld split: add one barycenter per original triangle, split into 3 triangles."""
    nnode = coords.shape[0]
    nelem = elems.shape[0]
    bary = coords[elems].mean(axis=1)
    coordsA = np.vstack([coords, bary])
    elemsA = np.zeros((3 * nelem, 3), dtype=int)
    parentA = np.zeros(3 * nelem, dtype=int)
    for K in range(nelem):
        a, b, c = elems[K]
        m = nnode + K
        r = 3 * K
        elemsA[r + 0] = [a, b, m]
        elemsA[r + 1] = [b, c, m]
        elemsA[r + 2] = [c, a, m]
        parentA[r:r+3] = K
    muA = np.r_[np.zeros(nnode), np.ones(nelem)]
    return coordsA, elemsA, parentA, muA


def build_vertex_to_elements(elems, nnode):
    v2e = [[] for _ in range(nnode)]
    for K, tri in enumerate(elems):
        for v in tri:
            v2e[v].append(K)
    return v2e


def local_z0_for_vertex(coords, elems, coordsA, elemsA, parentA, muA, v2e, y_id):
    nnode = coords.shape[0]
    nelemA = elemsA.shape[0]

    star_elems = np.array(v2e[y_id], dtype=int)
    if star_elems.size == 0:
        raise RuntimeError("isolated vertex")

    star_subtris = np.where(np.isin(parentA, star_elems))[0]
    local_nodes = np.unique(elems[star_elems].ravel())
    nloc = len(local_nodes)

    g2l = -np.ones(nnode, dtype=int)
    g2l[local_nodes] = np.arange(nloc)

    A = np.zeros((nloc, nloc))
    m = np.zeros(nloc)
    star_area = 0.0

    q = np.zeros(nloc)
    q[g2l[y_id]] = 1.0

    for K in star_elems:
        verts = elems[K]
        loc = g2l[verts]
        X = coords[verts]
        areaK, grads = tri_area_grads(X)
        star_area += areaK

        # int_K phi_i = |K|/3 for original P1 basis
        m[loc] += areaK / 3.0

        # exact int_K mu using its P1 representation on the three Alfeld subtriangles
        int_mu_K = 0.0
        for s in np.where(parentA == K)[0]:
            triA = elemsA[s]
            XA = coordsA[triA]
            areaSub, _ = tri_area_grads(XA)
            int_mu_K += areaSub * muA[triA].mean()

        A[np.ix_(loc, loc)] += int_mu_K * (grads @ grads.T)

    b = q - m / star_area

    # KKT solve: A c + m lambda = b, m^T c = 0.
    KKT = np.block([[A, m[:, None]], [m[None, :], np.zeros((1, 1))]])
    rhs = np.r_[b, 0.0]
    sol = np.linalg.solve(KKT, rhs)
    c = sol[:nloc]
    lam = sol[-1]

    ell_weak = m / star_area + A @ c

    # Build piecewise constant Z0 on each Alfeld subtriangle in the patch.
    z0_values = np.zeros(nelemA)
    for K in star_elems:
        verts = elems[K]
        loc = g2l[verts]
        X = coords[verts]
        _, grads = tri_area_grads(X)
        grad_psi = c[loc] @ grads
        for s in np.where(parentA == K)[0]:
            triA = elemsA[s]
            XA = coordsA[triA]
            _, gradsA = tri_area_grads(XA)
            grad_mu = muA[triA] @ gradsA
            z0_values[s] = 1.0 / star_area - np.dot(grad_mu, grad_psi)

    # Pointwise-Z0 local check against original P1 basis.
    ell_point = np.zeros(nloc)
    z0_integral = 0.0
    for K in star_elems:
        verts = elems[K]
        loc = g2l[verts]
        X = coords[verts]
        for s in np.where(parentA == K)[0]:
            triA = elemsA[s]
            XA = coordsA[triA]
            areaSub, _ = tri_area_grads(XA)
            lamA = barycentric_coords(XA, X)
            int_parent_phi = areaSub * lamA.mean(axis=0)
            ell_point[loc] += z0_values[s] * int_parent_phi
            z0_integral += z0_values[s] * areaSub

    checks = dict(
        sym_A=np.linalg.norm(A - A.T, ord="fro"),
        null_A=np.linalg.norm(A @ np.ones(nloc), ord=np.inf),
        rhs_compat=abs(b.sum()),
        zero_mean_psi=abs(m @ c),
        lambda_abs=abs(lam),
        repro_weak=np.linalg.norm(ell_weak - q, ord=np.inf),
        repro_pointwise=np.linalg.norm(ell_point - q, ord=np.inf),
        z0_integral=abs(z0_integral - 1.0),
    )

    return dict(
        y_id=y_id,
        star_elems=star_elems,
        star_subtris=star_subtris,
        local_nodes=local_nodes,
        A=A,
        m=m,
        b=b,
        q=q,
        psi_coeffs=c,
        lambda_value=lam,
        ell_weak=ell_weak,
        ell_point=ell_point,
        z0_values=z0_values,
        star_area=star_area,
        checks=checks,
    )


def build_pi0_matrix(coords, elems, coordsA, elemsA, parentA, muA):
    nnode = coords.shape[0]
    nelem = elems.shape[0]
    nnodeA = coordsA.shape[0]

    v2e = build_vertex_to_elements(elems, nnode)

    # E embeds original P1 nodal data into Alfeld P1 nodal data.
    rows = list(range(nnode))
    cols = list(range(nnode))
    vals = [1.0] * nnode
    for K in range(nelem):
        bary_id = nnode + K
        for v in elems[K]:
            rows.append(bary_id)
            cols.append(v)
            vals.append(1.0 / 3.0)
    E = sp.csr_matrix((vals, (rows, cols)), shape=(nnodeA, nnode))

    I, J, V = [], [], []
    all_checks = []
    outs = []

    for y_id in range(nnode):
        out = local_z0_for_vertex(coords, elems, coordsA, elemsA, parentA, muA, v2e, y_id)
        outs.append(out)
        all_checks.append(out["checks"])

        for s in out["star_subtris"]:
            triA = elemsA[s]
            XA = coordsA[triA]
            areaSub, _ = tri_area_grads(XA)
            val = out["z0_values"][s] * areaSub / 3.0
            for j in triA:
                I.append(y_id)
                J.append(j)
                V.append(val)

    P0 = sp.csr_matrix((V, (I, J)), shape=(nnode, nnodeA))

    data = {
        "max_local_weak_repro": max(c["repro_weak"] for c in all_checks),
        "max_local_point_repro": max(c["repro_pointwise"] for c in all_checks),
        "max_z0_integral_error": max(c["z0_integral"] for c in all_checks),
        "max_zero_mean_error": max(c["zero_mean_psi"] for c in all_checks),
        "max_null_error": max(c["null_A"] for c in all_checks),
        "locals": outs,
    }
    return P0, E, data


def run_demo(maxh=0.22, make_plots=True):
    ngmesh = unit_square.GenerateMesh(maxh=maxh)
    mesh = Mesh(ngmesh)
    coords, elems = extract_ngsolve_tri_mesh(mesh)
    coordsA, elemsA, parentA, muA = alfeld_split_2d(coords, elems)
    P0, E, data = build_pi0_matrix(coords, elems, coordsA, elemsA, parentA, muA)

    nnode = coords.shape[0]
    nnodeA = coordsA.shape[0]

    PEminusI = P0 @ E - sp.eye(nnode, format="csr")
    err_PE_entry = 0.0 if PEminusI.nnz == 0 else np.max(np.abs(PEminusI.data))
    err_PE_inf = sp.linalg.norm(PEminusI, np.inf)

    Q = E @ P0
    err_idempotent = sp.linalg.norm(Q @ Q - Q, np.inf)

    err_const = np.max(np.abs(P0 @ np.ones(nnodeA) - np.ones(nnode)))
    err_x = np.max(np.abs(P0 @ coordsA[:, 0] - coords[:, 0]))
    err_y = np.max(np.abs(P0 @ coordsA[:, 1] - coords[:, 1]))

    rng = np.random.default_rng(1)
    u0 = rng.standard_normal(nnode)
    err_random_p1 = np.max(np.abs(P0 @ (E @ u0) - u0))

    print("NGSolve / Netgen mesh")
    print(f"  original vertices        : {nnode}")
    print(f"  original triangles       : {elems.shape[0]}")
    print(f"  Alfeld vertices          : {nnodeA}")
    print(f"  Alfeld subtriangles      : {elemsA.shape[0]}")
    print("")
    print("Local Z0/psi checks over all vertices")
    print(f"  max local weak reproduction        : {data['max_local_weak_repro']:.3e}")
    print(f"  max local pointwise-Z0 reproduction: {data['max_local_point_repro']:.3e}")
    print(f"  max |int_st Z0 - 1|                : {data['max_z0_integral_error']:.3e}")
    print(f"  max |int_st psi|                   : {data['max_zero_mean_error']:.3e}")
    print(f"  max ||A*1||_inf                    : {data['max_null_error']:.3e}")
    print("")
    print("Global Pi0 matrix checks")
    print(f"  max entry |P0*E - I|               : {err_PE_entry:.3e}")
    print(f"  ||P0*E - I||_inf                   : {err_PE_inf:.3e}")
    print(f"  ||(E*P0)^2 - E*P0||_inf            : {err_idempotent:.3e}")
    print(f"  constant reproduction              : {err_const:.3e}")
    print(f"  x-coordinate reproduction          : {err_x:.3e}")
    print(f"  y-coordinate reproduction          : {err_y:.3e}")
    print(f"  random original-P1 reproduction    : {err_random_p1:.3e}")

    # Optional NGSolve GridFunction output. For first-order scalar H1 with no essential BC,
    # NGSolve has vertex-associated dofs; on this mesh ndof should equal number of vertices.
    Vh = H1(mesh, order=1)
    gfu = GridFunction(Vh)
    projected_values = P0 @ (np.sin(2*np.pi*coordsA[:,0]) * np.cos(np.pi*coordsA[:,1]) + coordsA[:,0]**2)
    if Vh.ndof == nnode:
        gfu.vec.FV().NumPy()[:] = projected_values
        print("")
        print("NGSolve GridFunction")
        print(f"  H1(order=1).ndof == nv             : {Vh.ndof == nnode}")
        print("  projected nodal values placed into GridFunction gfu")
    else:
        print("")
        print("NGSolve GridFunction")
        print(f"  H1(order=1).ndof = {Vh.ndof}, nv = {nnode}")
        print("  skipped direct vector assignment because dof ordering/size differs")

    if make_plots:
        plt.figure(figsize=(5,4))
        plt.triplot(coords[:,0], coords[:,1], elems, color="black", linewidth=0.8)
        plt.gca().set_aspect("equal")
        plt.title("Original NGSolve mesh")
        plt.tight_layout()

        plt.figure(figsize=(5,4))
        plt.spy(P0, markersize=2)
        plt.title("Sparsity of Pi0 matrix P0")
        plt.xlabel("Alfeld P1 input dof")
        plt.ylabel("original P1 output vertex")
        plt.tight_layout()

        plt.figure(figsize=(5,4))
        plt.tripcolor(coords[:,0], coords[:,1], elems, projected_values, shading="gouraud")
        plt.gca().set_aspect("equal")
        plt.colorbar()
        plt.title("Projected Pi0 u on original mesh")
        plt.tight_layout()

        plt.show()

    return mesh, coords, elems, coordsA, elemsA, P0, E, data


if __name__ == "__main__":
    run_demo(maxh=0.22, make_plots=False)
