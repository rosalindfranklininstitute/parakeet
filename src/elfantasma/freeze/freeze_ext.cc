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
#include <elfantasma/freeze/sphere_packer.h>

namespace py = pybind11;


PYBIND11_MODULE(freeze_ext, m)
{
  py::class_<elfantasma::SpherePacker>(m, "SpherePacker")
    .def(
      py::init<
        elfantasma::SpherePacker::grid_type,
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
    .def("index", &elfantasma::SpherePacker::index)
    .def("grid", &elfantasma::SpherePacker::grid)
    .def("node_length", &elfantasma::SpherePacker::node_length)
    .def("density", &elfantasma::SpherePacker::density)
    .def("radius", &elfantasma::SpherePacker::radius)
    .def("max_iter", &elfantasma::SpherePacker::max_iter)
    .def("multiplier", &elfantasma::SpherePacker::multiplier)
    .def("num_unplaced_samples", &elfantasma::SpherePacker::num_unplaced_samples)
    .def("next", &elfantasma::SpherePacker::next)
    .def("__len__", &elfantasma::SpherePacker::size)
    .def("__iter__", 
      [](elfantasma::SpherePacker &v) {
        return py::make_iterator(v.begin(), v.end());
      })
    ;

}
