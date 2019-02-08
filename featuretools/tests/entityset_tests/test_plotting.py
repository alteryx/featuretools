# -*- coding: utf-8 -*-

import os

import graphviz
import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


def test_returns_digraph_object(entityset):
    new_es = ft.EntitySet()

    graph = new_es.plot()

    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(entityset):
    new_es = ft.EntitySet()
    output_path = 'test1.png'

    new_es.plot(to_file=output_path)

    assert os.path.isfile(output_path)
    os.remove(output_path)


def test_missing_file_extension(entityset):
    new_es = ft.EntitySet()
    output_path = 'test1'

    with pytest.raises(ValueError) as excinfo:
        new_es.plot(to_file=output_path)

    assert str(excinfo.value).startswith('Please use a file extension')


def test_invalid_format(entityset):
    new_es = ft.EntitySet()
    output_path = 'test1.xzy'

    with pytest.raises(ValueError) as excinfo:
        new_es.plot(to_file=output_path)

    assert str(excinfo.value).startswith('Unknown format')
