Pi1 projector bundle v4
=======================

Files
-----
1. pi1projector.py
   Reusable Pi_0^1 implementation and validation-only helper routines.
   Public application API remains:

       P = Pi1Projector(mesh)
       Pi1_u = P(u)

   Validation helpers such as structured_unit_cube_six_tets,
   tetrahedral_hmax, ngmesh_from_data, and apply_pi0_for_validation live here
   so they do not clutter the notebooks.

2. Pi1_Projector_Colab_Demo.ipynb
   Clean user-facing application demo using an NGSolve/Netgen mesh.

3. Pi1_Projector_Validation.ipynb
   Numerical checks only: exact Nedelec reproduction, L2/curl errors,
   convergence rates, and the grad-Pi0/Pi1-grad commuting test.  Mesh/helper
   implementations are imported from pi1projector.py instead of being defined
   in the notebook.

Application demo mesh choices
-----------------------------
The application notebook includes three built-in Netgen/OCC examples:
  1. Fichera corner
  2. Hollow cylinder / coil
  3. Box with spherical cavity
Set `example = "fichera"`, `"coil"`, or `"cavity"`.
These examples intentionally do not call mesh.Curve(...).


Additional demo included in v6:
- Pi1_Projector_Colab_Demo_Load_VOL.ipynb : loads a user-supplied Netgen .vol mesh via Mesh("my_mesh.vol").


v7 sample Netgen mesh:
- sample_fichera.vol : native Netgen .vol tetrahedral Fichera-corner mesh (26 vertices, 42 tetrahedra).
- Pi1_Projector_Colab_Demo_Load_VOL.ipynb loads this file by default via Mesh("sample_fichera.vol").
