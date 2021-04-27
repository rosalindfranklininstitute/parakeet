/*
 *  freeze_ext.cc
 *
 *  Copyright (C) 2019 Diamond Light Source and Rosalond Franklin Institute
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <amplus/freeze/sphere_packer.h>

namespace py = pybind11;


PYBIND11_MODULE(amplus_ext, m)
{
  py::class_<amplus::SpherePacker>(m, "SpherePacker")
    .def(
      py::init<
        amplus::SpherePacker::grid_type,
        double,
        double,
        double,
        std::size_t,
        double>(),
          py::arg("grid"),
          py::arg("node_length"),
          py::arg("density"),
          py::arg("radius"),
          py::arg("max_iter") = 10,
          py::arg("multiplier") = 1.05)
    .def("index", &amplus::SpherePacker::index)
    .def("grid", &amplus::SpherePacker::grid)
    .def("node_length", &amplus::SpherePacker::node_length)
    .def("density", &amplus::SpherePacker::density)
    .def("radius", &amplus::SpherePacker::radius)
    .def("max_iter", &amplus::SpherePacker::max_iter)
    .def("multiplier", &amplus::SpherePacker::multiplier)
    .def("num_unplaced_samples", &amplus::SpherePacker::num_unplaced_samples)
    .def("next", &amplus::SpherePacker::next)
    .def("__len__", &amplus::SpherePacker::size)
    .def("__iter__", 
      [](amplus::SpherePacker &v) {
        return py::make_iterator(v.begin(), v.end());
      })
    ;

}
