import warnings
from packaging.version import parse
import logging
from featuretools.version import ENTITYSET_SCHEMA_VERSION, FEATURES_SCHEMA_VERSION

logger = logging.getLogger("featuretools.utils")


def check_schema_version(cls, cls_type):
    if isinstance(cls_type, str):
        current = None
        version_string = None
        if cls_type == "entityset":
            current = ENTITYSET_SCHEMA_VERSION
            version_string = cls.get("schema_version")
        elif cls_type == "features":
            current = FEATURES_SCHEMA_VERSION
            version_string = cls.features_dict["schema_version"]

        saved = version_string

        warning_text_upgrade = (
                "The schema version of the saved %s"
                "(%s) is greater than the latest supported (%s). "
                "You may need to upgrade featuretools. Attempting to load %s ..."
                % (cls_type, version_string, current, cls_type)
        )
        if parse(current) < parse(saved):
            warnings.warn(warning_text_upgrade)

        warning_text_outdated = (
                "The schema version of the saved %s"
                "(%s) is no longer supported by this version "
                "of featuretools. Attempting to load %s ..."
                % (cls_type, version_string, cls_type)
        )
        if parse(current).major > parse(saved).major:
            logger.warning(warning_text_outdated)
