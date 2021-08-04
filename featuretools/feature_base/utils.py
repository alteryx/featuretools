def is_valid_input(candidate, template):
    if template.logical_type is not None and not isinstance(candidate.logical_type, type(template.logical_type)):
        return False
    if template.semantic_tags - candidate.semantic_tags != set():
        return False
    return True
