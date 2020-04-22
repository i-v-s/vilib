/*
 * Class for holding a particular representation of an input image
 * subframe.cpp
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

#include <stdlib.h>
#include <string>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#ifdef ROS_SUPPORT
#include <sensor_msgs/image_encodings.h>
#endif /* ROS_SUPPORT */
#include "vilib/storage/subframe.h"
#include "vilib/storage/opencv.h"
#include "vilib/storage/ros.h"
#include "vilib/cuda_common.h"

namespace vilib {

Subframe::Subframe(std::size_t width,
                   std::size_t height,
                   std::size_t data_bytes,
                   Subframe::MemoryType type,
                   void *data,
                   std::size_t pitch) :
    cols(width), rows(height), data_bytes_(data_bytes), type_(type),
    pitch_(pitch ? pitch : width * data_bytes), data_(static_cast<unsigned char*>(data)),
    ownMemory(false)
{
    assert(pitch >= width * data_bytes);
}

Subframe::Subframe(std::size_t width,
                   std::size_t height,
                   std::size_t data_bytes,
                   MemoryType type) :
  cols(width), rows(height), data_bytes_(data_bytes), type_(type), ownMemory(true)
{
  // perform the memory allocations
  switch(type) {
    case MemoryType::PAGED_HOST_MEMORY: {
      total_bytes_ = width * data_bytes * height; // packed: width * height * data_bytes
      data_ = (unsigned char *)malloc(total_bytes_);
      pitch_ = width * data_bytes;
      break;
    }
    case MemoryType::PINNED_HOST_MEMORY: {
      total_bytes_ = width*height*data_bytes; // packed: width * height * data_bytes
      cudaMallocHost((void**)&data_, total_bytes_);
      pitch_ = width * data_bytes;
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY: {
      total_bytes_ = width*height*data_bytes; // packed: width * height * data_bytes
      cudaMalloc((void**)&data_, total_bytes_);
      pitch_ = width * data_bytes;
      break;
    }
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      cudaMallocPitch((void**)&data_, &pitch_, width * data_bytes, height);
      /*
       * Note to future self:
       * the returned pitch will be the calculated pitch in byte units
       */
      total_bytes_ = pitch_ * height;
      break;
    }
    case MemoryType::UNIFIED_MEMORY: {
      total_bytes_ = width * data_bytes * height; // packed: width * height * data_bytes
      cudaMallocManaged((void**)&data_,total_bytes_);
      pitch_ = width * data_bytes;
      break;
    }
  }
}

Subframe::Subframe(Subframe &&other) :
    cols(other.cols),
    rows(other.rows),
    data_bytes_(other.data_bytes_),
    type_(other.type_),
    total_bytes_(other.total_bytes_),
    pitch_(other.pitch_),
    data_(other.data_),
    ownMemory(other.ownMemory)
{
    other.data_ = nullptr;
    other.ownMemory = false;
}

Subframe::~Subframe(void) {
    // perform the memory deallocations
    if (ownMemory) switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
        free(data_);
        break;
    case MemoryType::PINNED_HOST_MEMORY:
        cudaFreeHost(data_);
        break;
    case MemoryType::LINEAR_DEVICE_MEMORY:
        /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY:
        /* fall-through */
    case MemoryType::UNIFIED_MEMORY:
        cudaFree(data_);
        break;
    }
}

Subframe &Subframe::operator =(Subframe &&other)
{
    cols = other.cols;
    rows = other.rows;
    data_bytes_ = other.data_bytes_;
    type_ = other.type_;
    total_bytes_ = other.total_bytes_;
    pitch_ = other.pitch_;
    data_ = other.data_;
    ownMemory = other.ownMemory;
    other.data_ = nullptr;
    other.ownMemory = false;
    return *this;
}

void Subframe::copy_from(const Subframe & h_img,
                         bool async,
                         cudaStream_t stream_num) {
    assert(h_img.cols == cols && h_img.rows == rows && h_img.data_bytes_ == data_bytes_);
    cudaMemcpyKind kind = cudaMemcpyDefault;
    switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
    case MemoryType::PINNED_HOST_MEMORY:
    case MemoryType::UNIFIED_MEMORY:
        switch(h_img.type_) {
        case MemoryType::PAGED_HOST_MEMORY:
        case MemoryType::PINNED_HOST_MEMORY:
        case MemoryType::UNIFIED_MEMORY:
            kind = cudaMemcpyHostToHost;
            break;
        case MemoryType::LINEAR_DEVICE_MEMORY:
        case MemoryType::PITCHED_DEVICE_MEMORY:
            kind = cudaMemcpyDeviceToHost;
            break;
        }
        break;
    case MemoryType::LINEAR_DEVICE_MEMORY:
    case MemoryType::PITCHED_DEVICE_MEMORY:
        switch(h_img.type_) {
        case MemoryType::PAGED_HOST_MEMORY:
        case MemoryType::PINNED_HOST_MEMORY:
        case MemoryType::UNIFIED_MEMORY:
            kind = cudaMemcpyHostToDevice;
            break;
        case MemoryType::LINEAR_DEVICE_MEMORY:
        case MemoryType::PITCHED_DEVICE_MEMORY:
            kind = cudaMemcpyDeviceToDevice;
            break;
        }
        break;
    }
    if(async)
        CUDA_API_CALL(cudaMemcpy2DAsync(data_, pitch_, h_img.data_, h_img.pitch_, cols, rows, kind, stream_num));
    else
        CUDA_API_CALL(cudaMemcpy2D(data_, pitch_, h_img.data_, h_img.pitch_, cols, rows, kind));
}


void Subframe::copy_from(const cv::Mat & h_img,
                         bool async,
                         cudaStream_t stream_num) {
  // TODO : support color!
  assert(h_img.channels() == 1);
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      opencv_copy_from_image_to_host(h_img,
                                     data_,
                                     pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      opencv_copy_from_image_to_gpu(h_img,
                                    data_,
                                    pitch_,
                                    async,
                                    stream_num);
      break;
    }
  }
}

#ifdef ROS_SUPPORT
void Subframe::copy_from(const sensor_msgs::ImageConstPtr & h_img,
                         bool async,
                         cudaStream_t stream_num) {
  // TODO: support color
  assert(h_img->encoding == sensor_msgs::image_encodings::MONO8);
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      ros_copy_from_image_to_host(h_img,
                                  1,
                                  data_,
                                  pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      ros_copy_from_image_to_gpu(h_img,
                                 1,
                                 data_,
                                 pitch_,
                                 async,
                                 stream_num);
      break;
    }
  }
}
#endif /* ROS_SUPPORT */

void Subframe::copy_to(cv::Mat & h_img,
                       bool async,
                       cudaStream_t stream_num) const {
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      opencv_copy_from_host_to_image(h_img,
                                     data_,
                                     cols,
                                     rows,
                                     data_bytes_,
                                     pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      opencv_copy_from_gpu_to_image(h_img,
                                    data_,
                                    cols,
                                    rows,
                                    data_bytes_,
                                    pitch_,
                                    async,
                                    stream_num);
      break;
    }
  }
}

vilib::Subframe::operator cv::Mat() const {
    cv::Mat result;
    copy_to(result);
    return result;
}

void Subframe::display(void) const {
    // copy image to a temporary buffer and display that
  std::string subframe_title("Subframe (");
  subframe_title += std::to_string(cols);
  subframe_title += "x";
  subframe_title += std::to_string(rows);
  subframe_title += ")";
  cv::Mat image;
  copy_to(image);
  cv::imshow(subframe_title.c_str(), image);
  cv::waitKey();
}

int Subframe::type() const
{
    switch(data_bytes_) {
    case 1: return CV_8UC1;
    case 2: return CV_8UC2;
    case 3: return CV_8UC3;
    case 4: return CV_8UC4;
    default:
        assert(!"Unknown type.");
        return 0;
    }
}

cv::Size Subframe::size() const
{
    return cv::Size(cols, rows);
}

} // namespace vilib
