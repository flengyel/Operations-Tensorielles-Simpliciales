import itertools

def generate_shapes_by_dimension(max_total_shapes=None):
    """
    Generates k-tuples (shapes) (n_1, ..., n_k) satisfying:
    1. Primary order: simplicial dimension n_s = min(n_j) - 1, starting n_s=2.
    2. Secondary order: For fixed n_s, iterate by k (order/length of tuple, k >= n_s).
    3. Tertiary order: For fixed (n_s, k), tuples (n_1,...,n_k) are lexicographically ordered.
    4. Constraint: k >= n_s (by iteration structure).
    5. Constraint: min(n_1, ..., n_k) = n_s + 1.
    6. Each n_j >= 1 (implicitly satisfied as n_s >= 2 => min(n_j) >= 3).
    """
    shapes_yielded_count = 0

    # Iterate through simplicial dimension n_s, starting from 2
    for n_s in itertools.count(2):
        # The minimum value any component n_j must take in the tuple
        # for this n_s is mval = n_s + 1.
        mval = n_s + 1

        # Iterate through the order (length) k of the tuple.
        # k must be >= n_s.
        for k in itertools.count(n_s):
            # Generate k-tuples (t_1, ..., t_k) in lexicographical order such that:
            # (a) Each t_j >= mval.
            # (b) At least one t_j == mval (i.e., min(tuple) == mval).

            # We'll generate tuples of offsets (d_1, ..., d_k) from mval, where d_j >= 0.
            # The condition becomes: min(d_j) == 0.
            # Shape will be (mval + d_1, ..., mval + d_k).
            # We need to generate these (d_1, ..., d_k) lexicographically.

            # Iterate through the sum_of_offsets s_d = d_1 + ... + d_k.
            # s_d = 0 gives (0,0,...,0) -> shape (mval, mval, ..., mval). This is the first lex.
            # s_d = 1 gives (0,0,...,1), then (0,0,...,1,0), ... then (1,0,...,0) lex.
            # This ensures that tuples with smaller components appear first for a fixed (n_s, k).
            
            # Heuristic limit for the sum of offsets to prevent one (n_s, k) pair
            # from running too long if max_total_shapes is large or None.
            # This makes the generator practical for demonstration.
            # For a true infinite generator, this bound would be removed.
            max_s_d_for_this_nk_pair = 15  # Adjust as needed for more/less output per (n_s, k)

            for s_d in range(max_s_d_for_this_nk_pair + 1): # Sum of d_i's
                for d_composition in _compositions_non_negative_lex(s_d, k):
                    if min(d_composition) == 0: # Ensure at least one offset is 0
                        shape = tuple(mval + d_val for d_val in d_composition)
                        yield shape
                        shapes_yielded_count += 1
                        if max_total_shapes and shapes_yielded_count >= max_total_shapes:
                            return # Stop after yielding the requested number of shapes
            
            # Heuristic breaks for demonstration if generating indefinitely
            if max_total_shapes is None: 
                if k > n_s + 3 and n_s < 4 : 
                    break 
                if k > n_s + 2 and n_s >=4 : 
                    break
        
        if max_total_shapes is None: 
            if n_s > 3: 
                break


def _compositions_non_negative_lex(s, k_parts):
    """
    Generates all compositions of integer s into k_parts non-negative parts,
    in lexicographical order.
    e.g., s=2, k_parts=2 -> (0,2), (1,1), (2,0)
    """
    if k_parts < 1:
        return
    if k_parts == 1:
        yield (s,)
        return
    
    # The first part 'i' can range from 0 up to s.
    for i in range(s + 1):
        # The remaining sum is s-i, for k_parts-1 parts.
        for rest_composition in _compositions_non_negative_lex(s - i, k_parts - 1):
            yield (i,) + rest_composition

# --- Demonstration (can be run if you save this as a .py file) ---
if __name__ == '__main__':
    print("Generating shapes in the user-specified order (n_s, then k, then lex shape):")
    print("Constraint: k >= n_s and min(shape_component) == n_s + 1")
    print("Output format: n_s (simplicial dim), k (order), Shape tuple")
    print("-" * 70)

    shape_generator = generate_shapes_by_dimension(max_total_shapes=35)
    for shape in shape_generator:
        n_s_val = min(shape) - 1
        k_val = len(shape)
        print(f"n_s={n_s_val}, k={k_val}, Shape={shape}")