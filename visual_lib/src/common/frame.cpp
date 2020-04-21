/*
 * Wrapper class for camera frames
 * frame.cpp
 *
 * Copyright (C) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef ROS_SUPPORT
#include <sensor_msgs/image_encodings.h>
#endif /* ROS_SUPPORT */
#include "vilib/common/frame.h"
#include "vilib/preprocess/image_preprocessing.h"
#include "vilib/preprocess/pyramid.h"
#include "vilib/config.h"

namespace vilib {

std::size_t Frame::last_id_ = 0;
std::mutex Frame::last_id_mutex_;

Frame::Frame(const cv::Mat & img,
             const int64_t timestamp_nsec,
             const std::size_t n_pyr_levels,
             cudaStream_t stream) :
    Frame(timestamp_nsec,
          img.cols,
          img.rows,
          n_pyr_levels) {
    preprocess_image(img, pyramid_, stream);
}

Frame::Frame(const Subframe & img,
             const int64_t timestamp_nsec,
             const std::size_t n_pyr_levels,
             cudaStream_t stream) :
    Frame(timestamp_nsec,
          img.cols,
          img.rows,
          n_pyr_levels) {
    preprocess_image(img, pyramid_, stream);
}


#ifdef ROS_SUPPORT
Frame::Frame(const sensor_msgs::ImageConstPtr & msg,
             const std::size_t n_pyr_levels,
             cudaStream_t stream) :
  Frame(msg->header.stamp.sec*1e9 + msg->header.stamp.nsec,
        msg->width,
        msg->height,
        n_pyr_levels) {
  preprocess_image(msg,pyramid_,stream);
}
#endif /* ROS_SUPPORT */

Frame::Frame(const int64_t timestamp_nsec,
             uint image_width,
             uint image_height,
             uint n_pyr_levels) :
    id_(getNewId()),
    timestamp_nsec_(timestamp_nsec),
    pyramid_pool(PyramidPool::get_pool(image_width, image_height, 1, n_pyr_levels, IMAGE_PYRAMID_MEMORY_TYPE))
  /*
   * Note: we allocate space for a grayscale image,
   *       irrespective of the input image
   */
{
    pyramid_pool->get(pyramid_);
}

std::size_t Frame::getNewId(void) {
  std::lock_guard<std::mutex> lock(last_id_mutex_);
  return last_id_++;
}

Frame::~Frame(void) {
    // return the pyramid buffers
    pyramid_pool->release(std::move(pyramid_));
}

image_pyramid_descriptor_t Frame::getPyramidDescriptor(void) const {
  image_pyramid_descriptor_t i;
  i.desc = pyramid_pool->get_descriptor();
  for(std::size_t l=0;l<pyramid_.size();++l) {
    i.data[l] = pyramid_[l].data_;
  }
  return i;
}

void Frame::resizeFeatureStorage(std::size_t new_size) {
  // Note: we dont want to lose features during this function call
  assert((int)new_size >= px_vec_.cols());
  // Don't do anything if the column size is the same
  if((int)new_size == px_vec_.cols()) {
    return;
  }
  std::size_t uninitialized_cols = new_size - num_features_;
  // do the resizing
  px_vec_.conservativeResize(Eigen::NoChange, new_size);
  level_vec_.conservativeResize(new_size, Eigen::NoChange);
  level_vec_.tail(uninitialized_cols).setZero();
  score_vec_.conservativeResize(new_size, Eigen::NoChange);
  track_id_vec_.conservativeResize(new_size);
  track_id_vec_.tail(uninitialized_cols).setConstant(-1);
}

} // namespace vilib
