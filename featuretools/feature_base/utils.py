def is_valid_input(candidate, template):
    """Checks if a candidate schema should be considered a match for a template schema"""
    if template.logical_type is not None and not isinstance(
        candidate.logical_type,
        type(template.logical_type),
    ):
        return False
    if len(template.semantic_tags - candidate.semantic_tags):
        return False
    return True
