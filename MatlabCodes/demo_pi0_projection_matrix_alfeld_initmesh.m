function demo_pi0_projection_matrix_alfeld_initmesh()
%DEMO_PI0_PROJECTION_MATRIX_ALFELD_INITMESH
%
% Builds the first projection matrix Pi_h^0 in 2D.
%
% Input representation:
%   continuous P1 nodal values on the Alfeld mesh.
%
% Output representation:
%   continuous P1 nodal values on the original mesh.
%
% Matrix:
%   P0(i,j) = int Z^0(y_i) Phi_j^A dx,
%
% where y_i is an original mesh vertex and Phi_j^A is an Alfeld P1 basis
% function.
%
% No essential boundary conditions are assumed. Boundary vertices are included.

close all;
clc;

%% ------------------------------------------------------------------------
% 1. Unit square mesh from initmesh
%% ------------------------------------------------------------------------

Hmax = 0.20; 
[p,e,t] = make_unit_square_initmesh(Hmax);

coords = p.';              % original nodes, nNode x 2
elems  = t(1:3,:).';       % original triangles, nElem x 3
elems  = orient_tris_ccw(coords, elems);

nNode = size(coords,1);
nElem = size(elems,1);

fprintf('\nOriginal mesh:\n');
fprintf('  nNode = %d\n', nNode);
fprintf('  nElem = %d\n', nElem);

%% ------------------------------------------------------------------------
% 2. Build global Alfeld split
%% ------------------------------------------------------------------------

[coordsA, elemsA, parentA, muA] = alfeld_split_2d(coords, elems);

nNodeA = size(coordsA,1);
nElemA = size(elemsA,1);

fprintf('\nAlfeld mesh:\n');
fprintf('  nNodeA = %d\n', nNodeA);
fprintf('  nElemA = %d\n', nElemA);

%% ------------------------------------------------------------------------
% 3. Build projection matrix P0
%% ------------------------------------------------------------------------

[P0, E, data] = build_pi0_matrix(coords, elems, coordsA, elemsA, parentA, muA);

fprintf('\nProjection matrix:\n');
fprintf('  size(P0) = %d x %d\n', size(P0,1), size(P0,2));
fprintf('  nnz(P0)  = %d\n', nnz(P0));

%% ------------------------------------------------------------------------
% 4. Checks
%% ------------------------------------------------------------------------

% E maps original P1 coefficients to Alfeld P1 coefficients.
% Therefore P0*E should be identity on original V_h^0.
PEminusI = P0*E - speye(nNode);

err_PE_entry = full(max(abs(PEminusI(:))));
err_PE_inf   = norm(PEminusI, inf);

% The square operator Q = E*P0 acts on Alfeld P1 nodal values.
% It should be a true projection: Q^2 = Q.
Q = E*P0;
err_idempotent = norm(Q*Q - Q, inf);

% Reproduce constants and coordinates.
err_const = max(abs(P0*ones(nNodeA,1) - ones(nNode,1)));
err_x     = max(abs(P0*coordsA(:,1) - coords(:,1)));
err_y     = max(abs(P0*coordsA(:,2) - coords(:,2)));

% Random original P1 function.
rng(1);
u0 = randn(nNode,1);
uA = E*u0;
u0_proj = P0*uA;
err_random_p1 = max(abs(u0_proj - u0));

fprintf('\nLocal row construction checks:\n');
fprintf('  max local weak reproduction error      = %.3e\n', data.max_local_weak_repro);
fprintf('  max local pointwise-Z0 reproduction    = %.3e\n', data.max_local_point_repro);
fprintf('  max abs(int_st Z0 - 1)                 = %.3e\n', data.max_z0_integral_error);
fprintf('  max abs(int_st psi)                    = %.3e\n', data.max_zero_mean_error);
fprintf('  max ||A*1||_inf                        = %.3e\n', data.max_null_error);

fprintf('\nGlobal projection checks:\n');
fprintf('  max entry abs(P0*E - I)                = %.3e\n', err_PE_entry);
fprintf('  matrix inf-norm ||P0*E - I||_inf       = %.3e\n', err_PE_inf);
fprintf('  idempotence ||(E*P0)^2 - E*P0||_inf    = %.3e\n', err_idempotent);
fprintf('  constant reproduction error            = %.3e\n', err_const);
fprintf('  x-coordinate reproduction error        = %.3e\n', err_x);
fprintf('  y-coordinate reproduction error        = %.3e\n', err_y);
fprintf('  random original-P1 reproduction error  = %.3e\n', err_random_p1);

fprintf('\nExpected result:\n');
fprintf('  P0*E should be identity up to roundoff.\n');
fprintf('  Constants, x, y, and random original P1 functions should reproduce.\n\n');

%% ------------------------------------------------------------------------
% 5. Example: project a non-original-P1 Alfeld function
%% ------------------------------------------------------------------------

uA_np = sin(2*pi*coordsA(:,1)).*cos(pi*coordsA(:,2)) + coordsA(:,1).^2;
u0_np = P0*uA_np;
uA_proj = E*u0_np;

figure;
triplot(elems, coords(:,1), coords(:,2), 'k');
axis equal tight;
title('Original mesh');
xlabel('x');
ylabel('y');

figure;
spy(P0);
title('Sparsity pattern of P0');
xlabel('Alfeld P1 input node');
ylabel('Original P1 output vertex');

figure;
patch('Faces', elemsA, ...
      'Vertices', coordsA, ...
      'FaceVertexCData', uA_np, ...
      'FaceColor', 'interp', ...
      'EdgeColor', [0.7 0.7 0.7]);
axis equal tight;
colorbar;
title('Input function on Alfeld mesh');
xlabel('x');
ylabel('y');

figure;
patch('Faces', elems, ...
      'Vertices', coords, ...
      'FaceVertexCData', u0_np, ...
      'FaceColor', 'interp', ...
      'EdgeColor', 'k');
axis equal tight;
colorbar;
title('Projected function: Pi_h^0 u in original P1 space');
xlabel('x');
ylabel('y');

figure;
patch('Faces', elemsA, ...
      'Vertices', coordsA, ...
      'FaceVertexCData', uA_proj, ...
      'FaceColor', 'interp', ...
      'EdgeColor', [0.7 0.7 0.7]);
axis equal tight;
colorbar;
title('Projected function embedded back onto Alfeld mesh');
xlabel('x');
ylabel('y');

    M = P0 * E;
    n = length(M);
    norm(M - speye(n,n),1)



end

%% ========================================================================
%  Build Pi_h^0 matrix
%% ========================================================================

function [P0, E, data] = build_pi0_matrix(coords, elems, coordsA, elemsA, parentA, muA)

nNode = size(coords,1);
nElem = size(elems,1);
nNodeA = size(coordsA,1);

% Embedding E : original P1 nodal values -> Alfeld P1 nodal values.
%
% Original vertices keep the same value.
% Each barycenter gets the average of the three original vertex values.
E = spalloc(nNodeA, nNode, nNode + 3*nElem);
E(1:nNode,1:nNode) = speye(nNode);

for K = 1:nElem
    bary_id = nNode + K;
    E(bary_id, elems(K,:)) = 1/3;
end

I = [];
J = [];
V = [];

max_local_weak_repro  = 0;
max_local_point_repro = 0;
max_z0_integral_error = 0;
max_zero_mean_error   = 0;
max_null_error        = 0;

% One row per original vertex, including boundary vertices.
for y_id = 1:nNode

    out = local_z0_for_vertex(coords, elems, coordsA, elemsA, parentA, muA, y_id);

    max_local_weak_repro  = max(max_local_weak_repro,  out.checks.repro_weak);
    max_local_point_repro = max(max_local_point_repro, out.checks.repro_pointwise);
    max_z0_integral_error = max(max_z0_integral_error, out.checks.z0_integral);
    max_zero_mean_error   = max(max_zero_mean_error,   out.checks.zero_mean_psi);
    max_null_error        = max(max_null_error,        out.checks.null_A);

    % Assemble row y_id:
    %
    %   P0(y_id,j) = int Z0(y_id) Phi_j^A dx.
    %
    % Z0 is piecewise constant on Alfeld subtriangles.
    for aa = 1:numel(out.star_subtris)

        s = out.star_subtris(aa);
        triA = elemsA(s,:);
        XA = coordsA(triA,:);

        [areaSub, ~] = tri_area_grads(XA);

        % Integral of each Alfeld P1 basis over this subtriangle is area/3.
        val = out.z0_values(s) * areaSub / 3;

        I = [I; y_id; y_id; y_id];
        J = [J; triA(:)];
        V = [V; val; val; val];
    end
end

P0 = sparse(I, J, V, nNode, nNodeA);

data.max_local_weak_repro  = max_local_weak_repro;
data.max_local_point_repro = max_local_point_repro;
data.max_z0_integral_error = max_z0_integral_error;
data.max_zero_mean_error   = max_zero_mean_error;
data.max_null_error        = max_null_error;

end

%% ========================================================================
%  Local Z0 and psi solve for one vertex
%% ========================================================================

function out = local_z0_for_vertex(coords, elems, coordsA, elemsA, parentA, muA, y_id)
%% ========================================================================
%  Local construction of Z^0(y) and psi_0(y) for one vertex
%% ========================================================================
%
% This routine is the local engine for one row of the Pi_h^0 projection
% matrix.
%
% Given one original mesh vertex y = y_id, it constructs the local weight
% function Z^0(y) used in the formula
%
%     (Pi_h^0 v)(y) = int_Omega Z^0(y) v dx.
%
% The construction follows
%
%     Z^0(y) = eta^0(y) - div(mu_y grad psi_0(y)),
%
% where
%
%     eta^0(y) = 1/|st(y)| on st(y), and zero outside,
%
% and st(y) is the vertex star, i.e. the union of all original triangles
% touching y.
%
% The unknown psi_0(y) is represented in the original continuous P1 basis
% restricted to st(y):
%
%     psi_0(y) = sum_j c_j phi_j.
%
% The coefficients c are computed from the local variational problem
%
%     int_st(y) mu_y grad psi_0(y) . grad u dx
%       = u(y) - (1/|st(y)|) int_st(y) u dx
%
% for all local P1 test functions u with zero mean on st(y). Since constants
% are in the kernel of the gradient form, the zero-mean condition
%
%     int_st(y) psi_0(y) dx = 0
%
% is imposed by a saddle-point system.
%
% The routine assembles:
%
%     A_ij = int_st(y) mu_y grad phi_j . grad phi_i dx,
%     m_i  = int_st(y) phi_i dx,
%     b_i  = phi_i(y) - m_i/|st(y)|.
%
% Then it solves
%
%     [ A   m ] [ c      ] = [ b ]
%     [ m'  0 ] [ lambda ]   [ 0 ]
%
% so that m' c = 0.
%
% After psi_0(y) is computed, the routine builds Z^0(y) explicitly as a
% piecewise constant function on the Alfeld subtriangles. On each Alfeld
% subtriangle, mu is affine and psi_0 is affine on the parent original
% triangle, so
%
%     div(mu grad psi_0) = grad(mu) . grad(psi_0).
%
% Hence
%
%     Z^0(y)|_{subtriangle}
%       = 1/|st(y)| - grad(mu) . grad(psi_0).
%
% The key correctness check is the reproduction of the vertex degree of
% freedom:
%
%     int_Omega Z^0(y) phi_i dx = phi_i(y)
%
% for every original P1 basis function phi_i supported on st(y). In vector
% form this is checked both weakly by
%
%     m/|st(y)| + A*c = q,
%
% and directly by integrating the piecewise-constant Z^0 values over the
% Alfeld subtriangles.
%
% The returned fields out.z0_values and out.star_subtris are used by the
% global assembly routine to form the row
%
%     P0(y_id,j) = int_Omega Z^0(y_id) Phi_j^A dx,
%
% where Phi_j^A is an Alfeld P1 basis function.
nNode = size(coords,1);
nElemA = size(elemsA,1);

% The vertex star is simply all original triangles containing y_id.
star_elems = find(any(elems == y_id, 2));
star_subtris = find(ismember(parentA, star_elems));

local_nodes = unique(elems(star_elems,:));
nloc = numel(local_nodes);

g2l = zeros(nNode,1);
g2l(local_nodes) = 1:nloc;

A = zeros(nloc,nloc);
m = zeros(nloc,1);
star_area = 0.0;

% q_i = phi_i(y)
q = zeros(nloc,1);
q(g2l(y_id)) = 1.0;

% Assemble
%
%   A_ij = int_st mu_y grad phi_j . grad phi_i dx,
%   m_i  = int_st phi_i dx.
%
% Here phi_i are original-mesh P1 basis functions restricted to st(y).
for kk = 1:numel(star_elems)

    K = star_elems(kk);
    verts = elems(K,:);
    loc = g2l(verts);

    X = coords(verts,:);
    [areaK, grads] = tri_area_grads(X);

    star_area = star_area + areaK;

    % int_K phi_i = |K|/3.
    m(loc) = m(loc) + areaK/3;

    % Exact integral int_K mu dx using the Alfeld subtriangles.
    sub_ids = find(parentA == K).';
    int_mu_K = 0.0;

    for s = sub_ids
        triA = elemsA(s,:);
        XA = coordsA(triA,:);
        mu_vals = muA(triA);

        [areaSub, ~] = tri_area_grads(XA);
        int_mu_K = int_mu_K + areaSub * mean(mu_vals);
    end

    % Since grad phi_i is constant on K:
    %
    %   int_K mu grad phi_j . grad phi_i
    %       = (int_K mu) grad phi_j . grad phi_i.
    A(loc,loc) = A(loc,loc) + int_mu_K * (grads * grads.');
end

% RHS:
%
%   b_i = phi_i(y) - (1/|st(y)|) int_st phi_i.
b = q - m/star_area;

% Solve singular Neumann-type system with zero-mean constraint:
%
%   A c + m lambda = b,
%   m^T c = 0.
KKT = [A, m; m.', 0];
rhs = [b; 0];

sol = KKT \ rhs;
c = sol(1:nloc);
lambda = sol(end);

% Weak action:
%
%   <Z0, phi_i> = (1/|st|) int_st phi_i
%                 + int_st mu grad psi . grad phi_i.
ell_weak = m/star_area + A*c;

% Construct pointwise Z0 on Alfeld subtriangles:
%
%   Z0 = 1/|st| - div(mu grad psi).
%
% On each Alfeld subtriangle, psi is affine on the parent original triangle,
% and mu is affine on the Alfeld subtriangle. Therefore
%
%   div(mu grad psi) = grad(mu) . grad(psi).
z0_values = zeros(nElemA,1);

for kk = 1:numel(star_elems)

    K = star_elems(kk);
    verts = elems(K,:);
    loc = g2l(verts);

    X = coords(verts,:);
    [~, grads] = tri_area_grads(X);

    grad_psi = c(loc).' * grads;

    sub_ids = find(parentA == K).';

    for s = sub_ids
        triA = elemsA(s,:);
        XA = coordsA(triA,:);
        mu_vals = muA(triA);

        [~, gradsA] = tri_area_grads(XA);

        grad_mu = mu_vals(:).' * gradsA;

        z0_values(s) = 1/star_area - dot(grad_mu, grad_psi);
    end
end

% Local pointwise-Z0 check against original P1 basis.
ell_point = zeros(nloc,1);
z0_integral = 0.0;

for kk = 1:numel(star_elems)

    K = star_elems(kk);
    verts = elems(K,:);
    loc = g2l(verts);
    X = coords(verts,:);

    sub_ids = find(parentA == K).';

    for s = sub_ids
        triA = elemsA(s,:);
        XA = coordsA(triA,:);

        [areaSub, ~] = tri_area_grads(XA);

        % Barycentric coordinates of Alfeld subtriangle vertices
        % with respect to parent original triangle K.
        lam = barycentric_coords(XA, X);

        % int_sub original_phi_a dx = areaSub * average(lambda_a).
        int_parent_phi = areaSub * mean(lam, 1);

        ell_point(loc) = ell_point(loc) + z0_values(s) * int_parent_phi(:);

        z0_integral = z0_integral + z0_values(s) * areaSub;
    end
end

checks.sym_A = norm(A - A.', 'fro');
checks.null_A = norm(A * ones(nloc,1), inf);
checks.rhs_compat = abs(sum(b));
checks.zero_mean_psi = abs(m.' * c);
checks.lambda = abs(lambda);
checks.repro_weak = norm(ell_weak - q, inf);
checks.repro_pointwise = norm(ell_point - q, inf);
checks.z0_integral = abs(z0_integral - 1.0);

out.star_elems = star_elems;
out.star_subtris = star_subtris;
out.local_nodes = local_nodes;
out.A = A;
out.m = m;
out.b = b;
out.q = q;
out.psi_coeffs = c;
out.lambda = lambda;
out.ell_weak = ell_weak;
out.ell_point = ell_point;
out.z0_values = z0_values;
out.star_area = star_area;
out.checks = checks;

end

%% ========================================================================
%  Mesh helpers
%% ========================================================================

function [p,e,t] = make_unit_square_initmesh(Hmax)

if exist('initmesh','file') ~= 2
    error(['initmesh was not found. This script requires the older PDE Toolbox ', ...
           'mesh functions initmesh/decsg.']);
end

% Unit square geometry:
%   (0,0), (1,0), (1,1), (0,1)
gd = [3; 4; 0; 1; 1; 0; 0; 0; 1; 1];

ns = char('S1');
ns = ns.';
sf = 'S1';

g = decsg(gd, sf, ns);

[p,e,t] = initmesh(g, 'Hmax', Hmax);

end

function elems = orient_tris_ccw(coords, elems)

for K = 1:size(elems,1)
    X = coords(elems(K,:),:);
    signed_area = 0.5 * det([X(2,:) - X(1,:); X(3,:) - X(1,:)]);
    if signed_area < 0
        elems(K,[2 3]) = elems(K,[3 2]);
    end
end

end

function [coordsA, elemsA, parentA, muA] = alfeld_split_2d(coords, elems)

nNode = size(coords,1);
nElem = size(elems,1);

bary = zeros(nElem,2);

for K = 1:nElem
    bary(K,:) = mean(coords(elems(K,:),:), 1);
end

coordsA = [coords; bary];

elemsA = zeros(3*nElem,3);
parentA = zeros(3*nElem,1);

for K = 1:nElem
    a = elems(K,1);
    b = elems(K,2);
    c = elems(K,3);
    m = nNode + K;

    r = 3*(K-1);

    elemsA(r+1,:) = [a, b, m];
    elemsA(r+2,:) = [b, c, m];
    elemsA(r+3,:) = [c, a, m];

    parentA(r+1:r+3) = K;
end

% Alfeld bubble:
%   mu = 0 at original vertices,
%   mu = 1 at each element barycenter.
muA = [zeros(nNode,1); ones(nElem,1)];

end

function [area, grads] = tri_area_grads(X)

% X is 3 x 2, counterclockwise.
% grads(i,:) = grad lambda_i.

x1 = X(1,1); y1 = X(1,2);
x2 = X(2,1); y2 = X(2,2);
x3 = X(3,1); y3 = X(3,2);

twoA = det([x2-x1, y2-y1; x3-x1, y3-y1]);
area = 0.5 * twoA;

if area <= 0
    error('Triangle has nonpositive area/orientation.');
end

grads = zeros(3,2);

grads(1,:) = [(y2-y3), (x3-x2)] / twoA;
grads(2,:) = [(y3-y1), (x1-x3)] / twoA;
grads(3,:) = [(y1-y2), (x2-x1)] / twoA;

end

function lam = barycentric_coords(P, X)

% Barycentric coordinates of points P with respect to triangle X.
%
% P: nPoint x 2
% X: 3 x 2
% lam: nPoint x 3

M = [X.'; ones(1,3)];
R = [P.'; ones(1,size(P,1))];

lam = (M \ R).';

end