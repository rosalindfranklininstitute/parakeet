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
#include <parakeet/freeze/sphere_packer.h>

namespace py = pybind11;


PYBIND11_MODULE(parakeet_ext, m)
{
  py::class_<parakeet::SpherePacker>(m, "SpherePacker")
    .def(
      py::init<
        parakeet::SpherePacker::grid_type,
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
    .def("index", &parakeet::SpherePacker::index)
    .def("grid", &parakeet::SpherePacker::grid)
    .def("node_length", &parakeet::SpherePacker::node_length)
    .def("density", &parakeet::SpherePacker::density)
    .def("radius", &parakeet::SpherePacker::radius)
    .def("max_iter", &parakeet::SpherePacker::max_iter)
    .def("multiplier", &parakeet::SpherePacker::multiplier)
    .def("num_unplaced_samples", &parakeet::SpherePacker::num_unplaced_samples)
    .def("next", &parakeet::SpherePacker::next)
    .def("__len__", &parakeet::SpherePacker::size)
    .def("__iter__", 
      [](parakeet::SpherePacker &v) {
        return py::make_iterator(v.begin(), v.end());
      })
    ;

}
