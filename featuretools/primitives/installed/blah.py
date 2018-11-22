from featuretools.primitive_utils.aggregation_primitive_base import (
    make_agg_primitive
)
from featuretools.variable_types import Index, Numeric, Variable


def count_func(values, count_null=False):
    if len(values) == 0:
        return 0

    if count_null:
        values = values.fillna(0)

    return values.count()


def count_generate_name(self):
    where_str = self._where_str()
    use_prev_str = self._use_prev_str()
    return u"COUNT(%s%s%s)" % (self.child_entity.id,
                               where_str,
                               use_prev_str)


Kanter = make_agg_primitive(count_func, [[Index], [Variable]], Numeric,
                            name="kanter_primitive", stack_on_self=False,
                            cls_attributes={"generate_name": count_generate_name})
