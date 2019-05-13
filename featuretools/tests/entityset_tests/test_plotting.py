# -*- coding: utf-8 -*-

import os

import graphviz
import pytest


def test_returns_digraph_object(es):
    graph = es.plot()

    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(es):
    output_path = 'test1.png'

    es.plot(to_file=output_path)

    assert os.path.isfile(output_path)
    os.remove(output_path)


def test_missing_file_extension(es):
    output_path = 'test1'

    with pytest.raises(ValueError) as excinfo:
        es.plot(to_file=output_path)

    assert str(excinfo.value).startswith('Please use a file extension')


def test_invalid_format(es):
    output_path = 'test1.xzy'

    with pytest.raises(ValueError) as excinfo:
        es.plot(to_file=output_path)

    assert str(excinfo.value).startswith('Unknown format')
