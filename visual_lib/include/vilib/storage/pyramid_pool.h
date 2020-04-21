/*
 * Class for handling the allocation of entire image pyramids efficiently
 * pyramid_pool.h
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

#pragma once

#include <assert.h>
#include <vector>
#include <map>
#include <mutex>
#include <forward_list>
#include <opencv2/core/mat.hpp>
#include "vilib/common/types.h"

namespace vilib {

class PyramidPool {
public:

    struct Params {
        uint original_width, original_height, data_bytes, levels;
        Subframe::MemoryType memory_type;
        bool operator <(const Params &other) const;
    };

    PyramidPool(const Params& params);

    static PyramidPool *get_pool(uint original_width, uint original_height, uint data_bytes, uint levels, Subframe::MemoryType memory_type)
    {
        Params d{original_width, original_height, data_bytes, levels, memory_type};
        auto it = pools.find(d);
        if (it == pools.end()) {
            return &pools.emplace(d, d).first->second;
        } else
            return &it->second;
    }

    /*
     * Acquire an entire image pyramid from a preallocated buffer. If the buffer
     * got empty, a new allocation will take place.
     * @param preallocated_item_num number of frames preallocated per pool
     * @param original_width pixel width of the level0 (highest resolution) image
     * @param original_height pixel height of the level0 (highest resolution) image
     * @param data_bytes number of databytes each pixel requires (e.g.: grayscale = 1)
     * @param levels number of levels in the pyramid
     * @param memory_type the type of memory that is used for each frame
     * @param pyramid destination vector holding the image pyramid
     */

    void get(std::vector<Subframe> & pyramid);

    /*
     * Return a previously acquired pyramid to the pyramid pool. The underlying
     * image frames will not be freed so that they can be reused.
     * @param pyramid the source vector holding the image pyramid
     */
    void release(std::vector<Subframe> && pyramid);


    /*
   * Return a pyramid descriptor with image widhts, heights and pitches
   * @return pyramid descriptor
   */
    inline pyramid_descriptor_t get_descriptor(void) {
        return desc_;
    }

    /*
   * Preallocate an entire image pyramid based on the level0 image width and
   * image height. One should not call this function directly.
   * @param preallocated_item_num number of frames preallocated per pool
   * @param original_width level0 image width
   * @param original_height level0 image height
   * @param data_bytes number of bytes each pixel requires
   * @param levels number of levels that get preallocated with decreasing resolutions
   *               each level halves the resolution of the preceding level
   * @param memory_type the type of memory that is used for each frame
   */
    void init(uint preallocated_item_num);

private:
    void create_pyramid(std::vector<Subframe> &pyramid);

    static std::map<Params, PyramidPool> pools;
    static std::mutex pools_mutex;

    std::forward_list<std::vector<Subframe>> items;
    std::size_t items_count;
    std::mutex items_mutex;
    std::size_t width_;
    std::size_t height_;
    std::size_t pitch_;
    std::size_t data_bytes_;
    Subframe::MemoryType type_;
    Params params;
    pyramid_descriptor_t desc_;
};

} // namespace vilib
