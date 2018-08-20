from psutil import virtual_memory


def mock_cluster(n_workers=1,
                 threads_per_worker=1,
                 diagnostics_port=8787,
                 memory_limit=None,
                 **dask_kwarg):
    return (n_workers, threads_per_worker, diagnostics_port, memory_limit)


class MockClient():
    def __init__(self, cluster):
        self.cluster = cluster

    def scheduler_info(self):
        return {'workers': {'worker 1': {'memory_limit': virtual_memory().total}}}
