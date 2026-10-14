# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Compatibility patches for PySCF integrations."""

from __future__ import annotations


def patch_pyscf_scf_opt() -> None:
    """Ensure PySCF SCF objects retain a valid `_opt` dict after deepcopy/unpickle.

    In PySCF, ``pyscf.scf.hf.SCF`` defines serialization methods via
    ``lib.generate_pickle_methods(excludes=('chkfile', '_chkfile', '_opt', '_eri', 'callback'))``.
    During ``__setstate__`` (invoked by :func:`pickle.loads` and :func:`copy.deepcopy`),
    excluded attributes are reset to ``None``.

    For small systems that fit in memory (``_is_mem_enough() == True``), PySCF computes
    2-electron integrals in-core and never accesses ``self._opt``. However, for larger systems
    (such as transition-metal clusters or large basis sets) requiring direct SCF
    (``_is_mem_enough() == False``), ``SCF.get_jk`` executes:

    .. code-block:: python

        if self.direct_scf and self._opt.get(omega) is None:
            ...

    If ``self._opt`` is ``None`` instead of a dictionary (its default in ``SCF.__init__`` is
    ``{None: None}``), this raises:
    ``AttributeError: 'NoneType' object has no attribute 'get'``.

    Because EmbASI deepcopies ASE calculator instances (which clone PySCF SCF objects) before
    running SCF steps (e.g., in ``AtomsEmbeddingASI.run_scf``), this patch guarantees that:
    1. ``SCF.__setstate__`` restores ``self._opt = {None: None}`` whenever ``_opt`` is ``None``.
    2. ``SCF.get_jk`` defensively re-initializes ``self._opt = {None: None}`` if ``_opt`` is ``None``.
    3. The ``SCF.opt`` property getter/setter safely handle ``self._opt is None``.
    """
    try:
        from pyscf.scf import hf
    except ImportError:
        return

    # 1. Patch SCF.__setstate__
    orig_setstate = getattr(hf.SCF, "__setstate__", None)
    if orig_setstate is not None and not getattr(hf.SCF, "_opt_setstate_patched", False):

        def _patched_setstate(self, state):
            orig_setstate(self, state)
            if getattr(self, "_opt", None) is None:
                self._opt = {None: None}

        hf.SCF.__setstate__ = _patched_setstate
        hf.SCF._opt_setstate_patched = True

    # 2. Defensively patch SCF.get_jk
    orig_get_jk = getattr(hf.SCF, "get_jk", None)
    if orig_get_jk is not None and not getattr(hf.SCF, "_opt_get_jk_patched", False):

        def _patched_get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
            if getattr(self, "_opt", None) is None:
                self._opt = {None: None}
            return orig_get_jk(
                self, mol=mol, dm=dm, hermi=hermi, with_j=with_j, with_k=with_k, omega=omega
            )

        hf.SCF.get_jk = _patched_get_jk
        hf.SCF._opt_get_jk_patched = True

    # 3. Patch SCF.opt property getter/setter defensively
    orig_opt_prop = getattr(hf.SCF, "opt", None)
    if isinstance(orig_opt_prop, property) and not getattr(hf.SCF, "_opt_prop_patched", False):

        def _opt_getter(self):
            if getattr(self, "_opt", None) is None:
                self._opt = {None: None}
            return self._opt[None]

        def _opt_setter(self, x):
            if getattr(self, "_opt", None) is None:
                self._opt = {None: None}
            self._opt[None] = x

        hf.SCF.opt = property(_opt_getter, _opt_setter)
        hf.SCF._opt_prop_patched = True


# Apply on module import
patch_pyscf_scf_opt()
