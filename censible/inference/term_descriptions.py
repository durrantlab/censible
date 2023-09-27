"""Utilities to process/interpret terms (e.g., atom-atom interactions)."""

import re


def get_numeric_val(s: str, varname: str) -> str:
    """Given a string and a variable name, return the value of that variable.
    
    Args:
        s (str): The string.
        varname (str): The variable name.
        
    Returns:
        str: The value of the variable.
    """
    # v is a number, so only digits, +/-, and .
    num_regex = "([e0-9\.\+\-]+)"
    v = re.search(f"{varname}={num_regex}", s)
    v = v.group(1) if v is not None else "?"
    return v


def full_term_description(term: str) -> str:
    """Given a term, return a more complete description.
    
    Args:
        term (str): The term.
        
    Returns:
        str: The more complete description.
    """
    # Given that making explainable scoring functions is the goal, good to
    # provide more complete description of the terms. This function tries to do
    # so semi-programatically.

    # TODO: Good to have David review this?

    desc = ""

    # Many terms share variables in common. Let's extract those first.
    o = get_numeric_val(term, "o")
    _w = get_numeric_val(term, "_w")
    _c = get_numeric_val(term, "_c")
    g = get_numeric_val(term, "g")
    _b = get_numeric_val(term, "_b")

    if term.startswith("atom_type_gaussian("):
        # Looks like atom_type_gaussian(t1=AliphaticCarbonXSHydrophobe,t2=AliphaticCarbonXSNonHydrophobe,o=0,_w=1,_c=8)
        # Extract t1, t2, o, _w, _c using regex.
        t1 = re.search("t1=(.*?),", term)
        t1 = t1[1] if t1 is not None else None

        t2 = re.search("t2=(.*?),", term)
        t2 = t2[1] if t2 is not None else None

        return f"adjacent atoms: {t1}-{t2}; offset: {o}; gaussian width: {_w}; distance cutoff: {_c}"
    elif term.startswith("gauss("):
        return f"sterics (vina): offset: {o}; gaussian width: {_w}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("repulsion("):
        return f"repulsion (vina): offset: {o}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("hydrophobic("):
        return f"hydrophobic (vina): good-distance cutoff: {g}; bad-distance cutoff: {_b}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("non_hydrophobic("):
        return f"non-hydrophobic: good-distance cutoff: {g}; bad-distance cutoff: {_b}; distance cutoff: {_c}; see ???"
    elif term.startswith("vdw("):
        i = get_numeric_val(term, "i")
        _j = get_numeric_val(term, "_j")
        _s = get_numeric_val(term, "_s")
        _ = get_numeric_val(term, "_\^")
        return f"vdw: Lennard-Jones exponents (AutoDock 4): {i}, {_j}; smoothing: {_s}; cap: {_}; distance cutoff: {_c}; see PMID17274016"
    elif term.startswith("non_dir_h_bond("):
        return f"non-directional hydrogen bond (vina): good-distance cutoff: {g}; bad-distance cutoff: {_b}; distance cutoff: {_c}; see PMC3041641"
    elif term.startswith("non_dir_anti_h_bond_quadratic"):
        return f"mimics repulsion between polar atoms that can't hydrogen bond: offset: {o}; distance cutoff: {_c}; see ???"
    elif term.startswith("non_dir_h_bond_lj("):
        _ = get_numeric_val(term, "_\^")
        return f"10-12 Lennard-Jones potential (AutoDock 4) : {o}; cap: {_}; distance cutoff: {_c}; see PMID17274016"
    elif term.startswith("acceptor_acceptor_quadratic("):
        return "quadratic potential (see repulsion) between two acceptor atoms: offset: {o}; distance cutoff: {_c}; see ???"
    elif term.startswith("donor_donor_quadratic("):
        return "quadratic potential (see repulsion) between two donor atoms: offset: {o}; distance cutoff: {_c}; see ???"
    elif term.startswith("ad4_solvation("):
        dsig = get_numeric_val(term, "d-sigma")
        sq = get_numeric_val(term, "_s/q")
        return f"desolvation (AutoDock 4): d-sigma: {dsig}; _s/q: {sq}; distance cutoff: {_c}; see PMID17274016"
    elif term.startswith("electrostatic("):
        i = get_numeric_val(term, "i")
        _ = get_numeric_val(term, "_\^")
        return f"electrostatics (AutoDock 4): distance exponent: {i}; cap: {_}; distance cutoff: {_c}; see PMID17274016"
    elif term == "num_heavy_atoms":
        return "number of heavy atoms"
    elif term == "num_tors_add":
        return "loss of torsional entropy upon binding (AutoDock 4); see PMID17274016"
    elif term == "num_hydrophobic_atoms":
        return "number of hydrophobic atoms"
    elif term == "ligand_length":
        return "length of the ligand"
    elif term in {"num_tors_sqr", "num_tors_sqrt"}:
        return "meaning uncertain"  # TODO: ask David
    else:
        return "error: unknown term"

