import logging
import warnings

from packaging.version import parse

from featuretools.version import ENTITYSET_SCHEMA_VERSION, FEATURES_SCHEMA_VERSION

logger = logging.getLogger("featuretools.utils")


def check_schema_version(cls, cls_type):
    """
    If the saved schema version is newer than the current featuretools
    schema version, this function will output a warning saying so.

    If the saved schema version is a major release or more behind
    the current featuretools schema version, this function will log
    a message saying so.
    """
    if isinstance(cls_type, str):
        current = None
        saved = None
        if cls_type == "entityset":
            current = ENTITYSET_SCHEMA_VERSION
            saved = cls.get("schema_version")
        elif cls_type == "features":
            current = FEATURES_SCHEMA_VERSION
            saved = cls.features_dict["schema_version"]

        if parse(current) < parse(saved):
            warning_text_upgrade = (
                "The schema version of the saved %s"
                "(%s) is greater than the latest supported (%s). "
                "You may need to upgrade featuretools. Attempting to load %s ..."
                % (cls_type, saved, current, cls_type)
            )
            warnings.warn(warning_text_upgrade)

        if parse(current).major > parse(saved).major:
            warning_text_outdated = (
                "The schema version of the saved %s"
                "(%s) is no longer supported by this version "
                "of featuretools. Attempting to load %s ..."
                % (cls_type, saved, cls_type)
            )
            logger.warning(warning_text_outdated)
