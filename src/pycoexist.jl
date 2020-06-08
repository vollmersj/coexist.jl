function pycoexist_setup()
    PyCall.pyimport_conda("pandas", "pandas")
    PyCall.pyimport_conda("numpy", "numpy")
    PyCall.pyimport_conda("scipy", "scipy")
    PyCall.pyimport_conda("dask", "dask")
    PyCall.pyimport_conda("cloudpickle", "cloudpickle")
    PyCall.pyimport_conda("distributed", "distributed")
    PyCall.pyimport_conda("xlrd", "xlrd")
end
