/*
 *  sphere_packer.h
 *
 *  Copyright (C) 2019 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the GPLv3 license, a copy of 
 *  which is included in the root directory of this package.
 */

#ifndef SPHERE_PACKER_H
#define SPHERE_PACKER_H

#include <array>
#include <cmath>
#include <iostream>
#include <list>
#include <random>
#include <vector>
#include <chrono>
#include <parakeet/error.h>

namespace parakeet {

  namespace detail {
    
    /**
     * Compute the norm
     * @param x: The vector
     * @returns: The vector norm
     */
    template <typename T, std::size_t N>
    T norm(const std::array<T,N> &x) {
      T result = 0;
      for (auto &xi : x) {
        result += xi*xi;
      }
      return std::sqrt(result);
    }
    
    /**
     * Compute the square distance between vectors
     * @param x: The first vector
     * @param y: The second vector
     * @returns: The square distance between the vectors
     */
    template <typename T, std::size_t N>
    T distance_sq(const std::array<T,N> &x, const std::array<T,N> &y) {
      T result = 0;
      for (std::size_t i = 0; i < N; ++i) {
        result += std::pow(x[i] - y[i], 2);
      }
      return result;
    }
    
    /**
     * Compute the distance between vectors
     * @param x: The first vector
     * @param y: The second vector
     * @returns: The distance between the vectors
     */
    template <typename T, std::size_t N>
    T distance(const std::array<T,N> &x, const std::array<T,N> &y) {
      return std::sqrt(distance_sq(x,y));
    }

  }

  /**
   * A class to generate uniformly distributed spheres in a volume
   *
   * Distributing hard non-overlapping spheres is a tricky problem. This class
   * aims to produce a uniform distribution of non overlapping spheres within
   * a given volume. In order to do this, the class makes use of the property 
   * of the uniform distribution that it is equivalent to 
   * a homogeneous spatial poisson process. Instead of computing uniformly
   * distributed points in the volume, we loop through nodes in a regular
   * grid and sample from a Poisson distribution with a given rate parameter.
   * This gives use the number of samples to place in this node. We then try
   * to place the spheres such that they don't overlap. By doing it this way
   * we only need to check spheres against their neighbours in the current
   * node and adjacent nodes rather than against all nodes. The position of the
   * sphere may be modified for a few iterations to try and make sure that is
   * does not overlap any of its neighbours.
   *
   * The resulting algorithm is much faster than testing all spheres against
   * every other sphere; however, producing many spheres in a large volume
   * is a time consuming process!
   */
  class SpherePacker {
  public:
    
    typedef std::array<std::size_t,3> grid_type;
    typedef std::array<double,3> coord_type;
    typedef std::list<coord_type> coord_list_type;
    typedef std::vector<coord_list_type> slice_type;
    typedef slice_type value_type;
    typedef slice_type& reference;
    typedef slice_type* pointer;
    typedef std::size_t size_type;

    /**
     * An iterator to provide a useful interace
     */
    class iterator {
    public:

      typedef std::forward_iterator_tag iterator_category;
      
      iterator(SpherePacker &packer, size_type slice_index)
        : packer_(packer),
          slice_index_(slice_index) {}
      
      iterator operator++() { 
        iterator it = *this; 
        AMPLUS_ASSERT(slice_index_ == packer_.index());
        slice_index_ = packer_.next();
        return it; 
      }
      
      iterator operator++(int) { 
        AMPLUS_ASSERT(slice_index_ == packer_.index());
        slice_index_ = packer_.next();
        return *this; 
      }
      
      reference operator*() {
        return packer_.slice();
      }
      
      pointer operator->() { 
        return &packer_.slice();
      }

      bool operator==(const iterator &other) {
        return slice_index_ == other.slice_index_;
      }

      bool operator!=(const iterator &other) {
        return !(*this == other);
      }

    private:
      SpherePacker& packer_;
      size_type slice_index_;
    };

    /**
     * Initialise the packer
     * @param grid: The grid size
     * @param node_length: The node length
     * @param density: The density of spheres per unit volume
     * @param radius: The radius of a sphere
     * @param max_iter: The number of iterations to relax the position
     * @param multiplier: The multiplier to move in each relax step
     */
    SpherePacker(
        grid_type grid, 
        double node_length, 
        double density, 
        double radius,
        std::size_t max_iter = 10,
        double multiplier = 1.05)
        : gen(std::random_device()()),
          grid_(grid),
          node_length_(node_length),
          density_(density),
          radius_(radius),
          max_iter_(max_iter),
          multiplier_(multiplier) {
     
      // Check the input 
      AMPLUS_ASSERT(grid[0] > 0 && grid[1] > 0 && grid[2] > 0);
      AMPLUS_ASSERT(node_length > 2*radius_);
      AMPLUS_ASSERT(density > 0);
      AMPLUS_ASSERT(radius > 0);
      AMPLUS_ASSERT(max_iter > 0);
      AMPLUS_ASSERT(multiplier > 0);

      // Compute the mean number of spheres per node (volume * density)
      node_rate_ = std::pow(node_length, 3) * density;

      // The total number of samples that couldn't be placed
      num_unplaced_samples_ = 0;

      // Initialise the slices
      std::size_t slice_size = grid[1] * grid[2];
      prev_slice_.resize(slice_size);
      curr_slice_.resize(slice_size);
      curr_index_ = 0;
    }

    /**
     * @returns The number of slices
     */
    size_type size() const { 
      return grid_[0]; 
    }

    /**
     * @returns The first iterator
     */
    iterator begin() {
      return iterator(*this, next());
    }

    /**
     * @returns The last iterator
     */
    iterator end() {
      return iterator(*this, grid_[0]);
    }

    /**
     * @returns The current index
     */
    size_type index() {
      return curr_index_;
    }

    /**
     * @returns The current slice
     */
    reference slice() {
      AMPLUS_ASSERT(index() != 0);
      return prev_slice_;
    }

    /**
     * @returns The grid
     */
    grid_type grid() const {
      return grid_;
    }

    /**
     * @returns The node length
     */
    double node_length() const {
      return node_length_;
    }

    /**
     * @returns The density 
     */
    double density() const {
      return density_;
    }

    /**
     * @returns The sphere radius
     */
    double radius() const {
      return radius_;
    }

    /**
     * @returns The maximum number of iterations
     */
    std::size_t max_iter() const {
      return max_iter_;
    }

    /**
     * @returns The multiplier
     */
    double multiplier() const {
      return multiplier_;
    }

    /**
     * @returns The number of samples that couldn't be placed
     */
    size_type num_unplaced_samples() const {
      return num_unplaced_samples_;
    }

    /**
     * Generate samples for the next slice
     * @returns The current index
     */
    size_type next() {
      
      // Check the index
      AMPLUS_ASSERT(index() < grid_[0]);

      // Setup the random number generators
      std::poisson_distribution<> poisson(node_rate_);
      std::uniform_real_distribution<> uniform(0, node_length_);

      // Get the current index
      std::size_t k = index();
      std::size_t additional_samples = 0;

      // Compute some bits beforehand. This is the closest allowed distance
      double closest_distance = 2 * radius_;
      double closest_distance_sq = std::pow(closest_distance,2);

      // Loop through all the nodes in this slice. This aim here is to uniformly
      // distribute spheres throughout the slice. To achieve this, we make use 
      // of the property of the uniform distribution that it is equivalent to
      // a homogeneous spatial poisson process. Instead of computing uniformly
      // distributed points in the volume, we loop through nodes in a regular
      // grid and sample from a Poisson distribution with a given rate parameter.
      // This gives use the number of samples to place in this node. We then try
      // to place the spheres such that they don't overlap. By doing it this way
      // we only need to check spheres against their neighbours in the current
      // node and adjacent nodes rather than against all nodes. 
      for (std::size_t j = 0, curr_node = 0; j < grid_[1]; ++j) {
        for (std::size_t i = 0; i < grid_[2]; ++i, ++curr_node) {

          // Compute the coordinates of the node corners
          double x0 = i * node_length_;
          double y0 = j * node_length_;
          double z0 = k * node_length_;
          double x1 = x0 + node_length_;
          double y1 = y0 + node_length_;
          double z1 = z0 + node_length_;

          // Compute the range of adjacent nodes to check
          std::size_t j0 = (j > 0 ? j-1 : 0);
          std::size_t j1 = (j < grid_[1] - 1 ? j+2 : grid_[1]);
          std::size_t i0 = (i > 0 ? i-1 : 0);
          std::size_t i1 = (i < grid_[2] - 1 ? i+2 : grid_[2]);

          // Sample from the poisson distribution
          std::size_t num_samples = poisson(gen);

          // Occassionally, if the number of samples is large and can't be fit
          // within a single node, there are some spheres that can't be fitted.
          // To deal with these, we store the number remaining at each node and
          // add these to the next node's allocation. Because some nodes will have
          // zero allocated, this seems to work ok.
          num_samples += additional_samples;

          // Loop through the number of samples we want to add
          for (std::size_t n = 0; n < num_samples; ++n) {
          
            // Generate a coordinate within the box
            coord_type p = { 
              uniform(gen) + x0, 
              uniform(gen) + y0, 
              uniform(gen) + z0
            };

            // Iterate a few times to place the sphere such that it doesn't
            // overlap with any sphere either in this node or in adjacent nodes
            for (std::size_t num_iter = 0; num_iter < max_iter_; ++num_iter) {
            
              // Loop through all the adjacent nodes (previous and current slice)
              // and check if the sphere at point p is overlapping with any other 
              // sphere. If the sphere overlaps then compute the gradient for a
              // 1/r^2 potential. 
              //
              // f = 1 / r^2 where r^2 = (p - c) . (p -c)
              // df / dx = -(p - c) / r^4  
              //
              // We could compute the potential for all points but this slows 
              // things down alot
              coord_type df = { 0, 0, 0 };
              double min_distance_sq = 2*closest_distance_sq;
              for (auto &slice : { &prev_slice_, &curr_slice_ }) {
                for (std::size_t jj = j0; jj < j1; ++jj) {
                  for (std::size_t ii = i0; ii < i1; ++ii) {
                    std::size_t node = ii + jj * grid_[2];
                    for (auto &c : (*slice)[node])  {
                      double distance_sq = detail::distance_sq(c, p);
                      if (distance_sq < closest_distance_sq) {
                        double distance_m4 = 1.0 / std::pow(distance_sq, 2);
                        df[0] -= distance_m4 * (p[0] - c[0]);
                        df[1] -= distance_m4 * (p[1] - c[1]);
                        df[2] -= distance_m4 * (p[2] - c[2]);
                        min_distance_sq = std::min(min_distance_sq, distance_sq);
                      }
                    }
                  }
                }
              }

              // If the sphere is not overlapping then add it to the node
              // Otherwise, compute the step position to move the node to a new
              // position based on the computed gradient. We don't want to find
              // the minimum so we only move a distance along the gradient that
              // will hopefully put us just outside the circle. If we have moved
              // outside the node then break and discard the point.
              if (min_distance_sq >= closest_distance_sq) {
                AMPLUS_ASSERT(p[0] >= x0 && p[0] < x1);
                AMPLUS_ASSERT(p[1] >= y0 && p[1] < y1);
                AMPLUS_ASSERT(p[2] >= z0 && p[2] < z1);
                curr_slice_[curr_node].push_back(p);
                break;
              } else {
                double norm = detail::norm(df);
                double min_distance = std::sqrt(min_distance_sq);
                double step = multiplier_ * (closest_distance - min_distance) / norm;
                p[0] -= step * df[0];
                p[1] -= step * df[1];
                p[2] -= step * df[2];
                if (p[0] < x0 || p[0] > x1 ||
                    p[1] < y0 || p[1] > y1 ||
                    p[2] < z0 || p[2] > z1) {
                  break;
                }
              }
            }
          }

          // If some samples haven't beed added then keep them for the next node
          additional_samples = num_samples - curr_slice_[curr_node].size();
        }
      }

      // Swap the current and previous slices
      prev_slice_.swap(curr_slice_);
      curr_slice_.clear();
      curr_slice_.resize(prev_slice_.size());

      // Add the number of unplaced samples
      num_unplaced_samples_ += additional_samples;

      // Return the next index
      return ++curr_index_;
    }

  private:
    std::mt19937 gen;
    grid_type grid_; 
    double node_length_;
    double density_;
    double radius_;
    std::size_t max_iter_;
    double multiplier_;
    double node_rate_;
    slice_type prev_slice_;
    slice_type curr_slice_;
    size_type curr_index_;
    size_type num_unplaced_samples_;
  };

}

#endif // SPHERE_PACKER_H
