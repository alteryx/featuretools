# -*- coding: utf-8 -*-

import os

import graphviz
import pytest

from ..testing_utils import make_ecommerce_entityset


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_returns_digraph_object(entityset):
    graph = entityset.plot()

    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(entityset):
    output_path = 'test1.png'

    entityset.plot(to_file=output_path)

    assert os.path.isfile(output_path)
    os.remove(output_path)


def test_missing_file_extension(entityset):
    output_path = 'test1'

    with pytest.raises(ValueError) as excinfo:
        entityset.plot(to_file=output_path)

    assert str(excinfo.value).startswith('Please use a file extension')


def test_invalid_format(entityset):
    output_path = 'test1.xzy'

    with pytest.raises(ValueError) as excinfo:
        entityset.plot(to_file=output_path)

    assert str(excinfo.value).startswith('Unknown format')
