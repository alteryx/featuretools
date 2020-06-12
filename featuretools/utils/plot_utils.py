from featuretools.utils.gen_utils import import_or_raise


def check_graphviz():
    GRAPHVIZ_ERR_MSG = ('Please install graphviz to plot.' +
                        ' (See https://docs.featuretools.com/en/stable/getting_started/install.html#installing-graphviz for' +
                        ' details)')
    graphviz = import_or_raise("graphviz", GRAPHVIZ_ERR_MSG)
    # Try rendering a dummy graph to see if a working backend is installed
    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        raise RuntimeError(
            "To plot entity sets, a graphviz backend is required.\n" +
            "Install the backend using one of the following commands:\n" +
            "  Mac OS: brew install graphviz\n" +
            "  Linux (Ubuntu): sudo apt-get install graphviz\n" +
            "  Windows: conda install python-graphviz\n" +
            "  For more details visit: https://docs.featuretools.com/en/stable/getting_started/install.html"
        )
    return graphviz


def get_graphviz_format(graphviz, to_file):
    if to_file:
        # Explicitly cast to str in case a Path object was passed in
        to_file = str(to_file)

        split_path = to_file.split('.')
        if len(split_path) < 2:
            raise ValueError("Please use a file extension like '.pdf'" +
                             " so that the format can be inferred")

        format_ = split_path[-1]
        valid_formats = graphviz.backend.FORMATS
        if format_ not in valid_formats:
            raise ValueError("Unknown format. Make sure your format is" +
                             " amongst the following: %s" % valid_formats)
    else:
        format_ = None
    return format_


def save_graph(graph, to_file, format_):
    # Graphviz always appends the format to the file name, so we need to
    # remove it manually to avoid file names like 'file_name.pdf.pdf'
    offset = len(format_) + 1  # Add 1 for the dot
    output_path = to_file[:-offset]
    graph.render(output_path, cleanup=True)
