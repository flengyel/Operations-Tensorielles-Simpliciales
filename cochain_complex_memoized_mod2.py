import sympy as sp
import itertools
import numpy as np

class CochainComplex:
    def __init__(self, shape):
        self.shape = shape
        self.order = len(shape)
        self.generators = {}  # degree → list of (index, symbol)
        self.boundary_matrices = {}  # degree → boundary matrix
        self._build_generators()
        self._build_boundary_matrices()

    def _build_generators(self):
        index_range = [range(s) for s in self.shape]
        for n in range(self.order + 2):
            terms = []
            for idx in itertools.product(*index_range):
                if len(set(idx)) >= n:
                    symbol = sp.Symbol(f"x_{{{','.join(map(str, idx))}}}")
                    terms.append((idx, symbol))
            self.generators[n] = terms

    def _build_boundary_matrices(self):
        for degree in range(1, self.order + 2):
            self._build_boundary_matrix(degree)

    def _build_boundary_matrix(self, degree):
        source = [sym for idx, sym in self.generators.get(degree, [])]
        target = [sym for idx, sym in self.generators.get(degree - 1, [])]

        if not source or not target:
            self.boundary_matrices[degree] = sp.zeros(len(target), len(source))
            return

        matrix_entries = []
        for target_idx, target_sym in self.generators[degree - 1]:
            row = []
            for source_idx, source_sym in self.generators[degree]:
                coeff = self._compute_boundary_coefficient(source_idx, target_idx)
                row.append(coeff)
            matrix_entries.append(row)

        self.boundary_matrices[degree] = sp.Matrix(matrix_entries)

    def _compute_boundary_coefficient(self, source_idx, target_idx):
        if len(target_idx) + 1 != len(source_idx):
            return 0
        for i in range(len(source_idx)):
            face = source_idx[:i] + source_idx[i + 1:]
            if face == target_idx:
                return (-1) ** i
        return 0

    def get_symbol(self, idx):
        return sp.Symbol(f"x_{{{','.join(map(str, idx))}}}")

    def get_degree(self, expr):
        terms = expr.as_ordered_terms()
        if not terms:
            return None
        for term in terms:
            if isinstance(term, sp.Symbol):
                sym = term
            else:
                sym_parts = [arg for arg in term.args if isinstance(arg, sp.Symbol)]
                if not sym_parts:
                    continue
                sym = sym_parts[0]
            sym_str = sym.name
            if sym_str.startswith("x_{") and sym_str.endswith("}"):
                idx_str = sym_str[2:-1]
                try:
                    idx = tuple(map(int, idx_str.split(",")))
                    return len(idx)
                except ValueError:
                    continue
        return None

    def boundary(self, expr):
        degree = self.get_degree(expr)
        if degree is None or degree not in self.boundary_matrices:
            return sp.S.Zero

        vec = self.flatten(expr, degree)
        result = self.boundary_matrices[degree] @ vec
        expr_result = sp.S.Zero
        for i, coeff in enumerate(result):
            if coeff != 0:
                _, sym = self.generators[degree - 1][i]
                expr_result += coeff * sym
        return expr_result

    def coboundary(self, expr):
        degree = self.get_degree(expr)
        if degree is None or degree + 1 not in self.boundary_matrices:
            return sp.S.Zero

        vec = self.flatten(expr, degree)
        result = self.boundary_matrices[degree + 1].T @ vec
        expr_result = sp.S.Zero
        for i, coeff in enumerate(result):
            if coeff != 0:
                _, sym = self.generators[degree + 1][i]
                expr_result += coeff * sym
        return expr_result

    def flatten(self, expr, degree):
        symbols = [sym for idx, sym in self.generators[degree]]
        coeffs = [expr.coeff(sym) for sym in symbols]
        return sp.Matrix(coeffs)

    def is_cocycle(self, expr):
        cob = self.coboundary(expr)
        return sp.simplify(cob) == 0

    def is_coboundary(self, expr):
        degree = self.get_degree(expr)
        if degree is None or degree == 0:
            return expr == 0

        prev_matrix = self.boundary_matrices.get(degree)
        if prev_matrix is None:
            return False

        vec = self.flatten(expr, degree)
        try:
            sol = prev_matrix.gauss_jordan_solve(vec)[0]
            residual = prev_matrix @ sol - vec
            is_solution = all(c.is_zero for c in residual)
            return bool(is_solution)
        except Exception:
            return False

if __name__ == "__main__":
    complex = CochainComplex(shape=(3, 3))

    exprs = [
        sp.Symbol("x_{0,1}") - sp.Symbol("x_{1,1}") + sp.Symbol("x_{1,2}"),
        sp.Symbol("x_{1,0}") - sp.Symbol("x_{1,1}") + sp.Symbol("x_{2,1}"),
        - sp.Symbol("x_{1,1}") + sp.Symbol("x_{1,2}")

    ]

    for i, filler_expr in enumerate(exprs, start=1):
        degree = complex.get_degree(filler_expr)
        print(f"Filler {i} (Degree {degree}):", filler_expr)
        print("Is cocycle:", complex.is_cocycle(filler_expr))
        print("Is coboundary:", complex.is_coboundary(filler_expr))
        print()


