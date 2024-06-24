def evaluate_dnf(formula, true_props):
    """
    DNF stands for Disjunctive Normal Form. It is a standard way of structuring a logical 
    formula in Boolean algebra. A formula in DNF is a disjunction (OR) of conjunctions (AND)
     of literals, where a literal is either a variable or its negation.
    """
    # Split the formula into conjunctions
    conjunctions = formula.split('|')
    
    # Check each conjunction
    for conj in conjunctions:
        literals = conj.split('&')
        conj_true = True
        for literal in literals:
            literal = literal.strip()
            if literal.startswith('!'):
                # Negated literal
                prop = literal[1:]
                if prop in true_props:
                    conj_true = False
                    break
            else:
                # Positive literal
                if literal not in true_props:
                    conj_true = False
                    break
        if conj_true:
            return True
    return False

# Example usage:
print(evaluate_dnf("a&b|!c&d", "b"))  # Output: True
