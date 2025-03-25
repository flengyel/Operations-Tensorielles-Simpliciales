#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    cochain_complex.py
#
#    Copyright (C) 2021-2025 Florian Lengyel
#    Email: florian.lengyel at cuny edu, florian.lengyel at gmail
#    Website: https://github.com/flengyel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sympy as sp
import itertools

class CochainComplex:
    def __init__(self, shape):
        self.shape = shape
        self.order = len(shape)
        self.generators = {}  # degree → list of (index, symbol)
        self._build_generators()

    def _build_generators(self):
        index_range = [range(s) for s in self.shape]
        for n in range(1, self.order + 2):
            terms = []
            for idx in itertools.product(*index_range):
                if len(set(idx)) >= n:
                    symbol = sp.Symbol(f"x_{{{','.join(map(str, idx))}}}")
                    terms.append((idx, symbol))
            self.generators[n - 1] = terms

    def get_symbol(self, idx):
        return sp.Symbol(f"x_{{{','.join(map(str, idx))}}}")

    def get_degree(self, expr):
        """
        Get the degree of a symbolic expression based on the number of indices in its terms.
        """
        terms = expr.as_ordered_terms()
        if not terms:
            return None
            
        for term in terms:
            # Handle both standalone symbols and coefficients*symbols
            if isinstance(term, sp.Symbol):
                sym = term
            else:
                # Extract the symbol part from a Mul expression
                sym_parts = [arg for arg in term.args if isinstance(arg, sp.Symbol)]
                if not sym_parts:
                    continue
                sym = sym_parts[0]
            
            sym_str = sym.name
            if sym_str.startswith("x_{") and sym_str.endswith("}"):
                idx_str = sym_str[3:-1]  # strip x_{ and }
                try:
                    idx = tuple(map(int, idx_str.split(",")))
                    return len(idx)
                except ValueError:
                    continue
        
        return None

    def delta(self, expr):
        """
        Coboundary operator δ applied to a linear combination of generators.
        """
        if isinstance(expr, sp.Symbol):
            expr = sp.Add(expr)

        result = 0
        for term in expr.as_ordered_terms():
            coeff, sym = term.as_coeff_Mul()
            sym_str = str(sym)
            if not sym_str.startswith("x_{") or not sym_str.endswith("}"):
                continue  # skip malformed symbols
            idx_str = sym_str[2:-1]  # strip x_{ and }
            try:
                idx = tuple(map(int, idx_str.split(",")))
            except ValueError:
                continue
            k = len(idx)
            for i in range(k):
                face = idx[:i] + idx[i+1:]  # remove i-th entry
                if len(face) != self.order:
                    continue
                if all(0 <= face[j] < self.shape[j] for j in range(len(face))):
                    sign = (-1) ** i
                    face_sym = self.get_symbol(face)
                    result += coeff * sign * face_sym
        return sp.simplify(result)

    def is_cocycle(self, expr):
        return self.delta(expr) == 0

    def is_coboundary(self, expr):
        terms = expr.as_ordered_terms()
        if not terms:
            return False
        any_term = terms[0]
        sym_str = str(any_term.args[-1]) if any_term.args else str(any_term)
        if not sym_str.startswith("x_{") or not sym_str.endswith("}"):
            return False
        idx_str = sym_str[2:-1]  # strip x_{ and }
        try:
            degree = len(tuple(map(int, idx_str.split(","))))
        except ValueError:
            return False

        candidate_degree = degree - 1
        if candidate_degree not in self.generators:
            return False

        image = set()
        for idx, sym in self.generators[candidate_degree]:
            image.add(self.delta(sym))
        return any(sp.simplify(expr - cob) == 0 for cob in image)

if __name__ == "__main__":
    complex = CochainComplex(shape=(3, 3))

    exprs = [
        sp.Symbol("x_{0,1}") - sp.Symbol("x_{1,1}") + sp.Symbol("x_{1,2}"),
        sp.Symbol("x_{1,0}") - sp.Symbol("x_{1,1}") + sp.Symbol("x_{2,1}")
    ]

    for i, filler_expr in enumerate(exprs, start=1):
        degree = complex.get_degree(filler_expr)
        print(f"Filler {i} (Degree {degree}):", filler_expr)
        print("Is cocycle:", complex.is_cocycle(filler_expr))
        print("Is coboundary:", complex.is_coboundary(filler_expr))
        print()

    complex = CochainComplex(shape=(4, 4, 4))

    exprs = [
        sp.Symbol("x_{0,1,2}") - sp.Symbol("x_{0,2,2}") + sp.Symbol("x_{0,2,3}") - sp.Symbol("x_{1,1,2}") + sp.Symbol("x_{1,1,3}") + sp.Symbol("x_{1,2,2}") - sp.Symbol("x_{1,2,3}"),
        sp.Symbol("x_{0,2,1}") - sp.Symbol("x_{0,2,2}") + sp.Symbol("x_{0,3,2}") - sp.Symbol("x_{1,2,1}") + sp.Symbol("x_{1,2,2}") + sp.Symbol("x_{1,3,1}") - sp.Symbol("x_{1,3,2}"),
        sp.Symbol("x_{1,0,2}") - sp.Symbol("x_{1,1,2}") + sp.Symbol("x_{1,1,3}") - sp.Symbol("x_{2,0,2}") + sp.Symbol("x_{2,0,3}") + sp.Symbol("x_{2,1,2}") - sp.Symbol("x_{2,1,3}"),
        sp.Symbol("x_{1,2,0}") - sp.Symbol("x_{1,2,1}") + sp.Symbol("x_{1,3,1}") - sp.Symbol("x_{2,2,0}") + sp.Symbol("x_{2,2,1}") + sp.Symbol("x_{2,3,0}") - sp.Symbol("x_{2,3,1}"),
        sp.Symbol("x_{2,0,1}") - sp.Symbol("x_{2,0,2}") - sp.Symbol("x_{2,1,1}") + sp.Symbol("x_{2,1,2}") + sp.Symbol("x_{3,0,2}") + sp.Symbol("x_{3,1,1}") - sp.Symbol("x_{3,1,2}"),
        sp.Symbol("x_{2,1,0}") - sp.Symbol("x_{2,1,1}") - sp.Symbol("x_{2,2,0}") + sp.Symbol("x_{2,2,1}") + sp.Symbol("x_{3,1,1}") + sp.Symbol("x_{3,2,0}") - sp.Symbol("x_{3,2,1}")
    ]

    for i, filler_expr in enumerate(exprs, start=1):
        degree = complex.get_degree(filler_expr)
        print(f"Filler {i} (Degree {degree}):", filler_expr)
        print("Is cocycle:", complex.is_cocycle(filler_expr))
        print("Is coboundary:", complex.is_coboundary(filler_expr))
        print()

