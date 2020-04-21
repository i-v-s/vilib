/*
 * Class for handling the allocation of entire image pyramids efficiently
 * pyramid_pool.cpp
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

#include <assert.h>
#include <mutex>
#include "vilib/storage/pyramid_pool.h"
#include "vilib/storage/subframe.h"

namespace vilib {

std::map<PyramidPool::Params, PyramidPool> PyramidPool::pools;
std::mutex PyramidPool::pools_mutex;

void PyramidPool::init(uint preallocated_item_num) {
    assert(params.levels > 0);
    assert((params.original_width  % (1 << (params.levels - 1))) == 0);
    assert((params.original_height % (1 << (params.levels - 1))) == 0);

    for (uint i = 0; i < preallocated_item_num; ++i)
        create_pyramid(items.emplace_front());
}


void PyramidPool::create_pyramid(std::vector<Subframe> &pyramid)
{
    pyramid.reserve(params.levels);
    for (uint l = 0, w = params.original_width, h = params.original_height; l < params.levels; ++l) {
        pyramid.emplace_back(w, h, params.data_bytes, params.memory_type);
        w /= 2;
        h /= 2;
    }
}

PyramidPool::PyramidPool(const Params &params) :
    params(params)
{
    init(IMAGE_PYRAMID_PREALLOCATION_ITEM_NUM);
    for (uint l = 0, w = params.original_width, h = params.original_height; l < params.levels; ++l) {
        desc_.w[l] = w;
        desc_.h[l] = h;
        desc_.p[l] = items.front()[l].pitch_;
        w /= 2;
        h /= 2;
    }
    desc_.l = params.levels;
}

void PyramidPool::get(std::vector<Subframe> &pyramid)
{
    if (items.empty())
        create_pyramid(pyramid);
    else {
        std::lock_guard<std::mutex> lock(items_mutex);
        pyramid = std::move(items.front());
        items.pop_front();
        --items_count;
    }
}

void PyramidPool::release(std::vector<Subframe> && pyramid) {
    std::lock_guard<std::mutex> lock(items_mutex);
    items.emplace_front(std::move(pyramid));
    ++items_count;
}

bool PyramidPool::Params::operator <(const PyramidPool::Params &other) const
{
    if (other.original_height != original_height) return original_height < other.original_height;
    if (other.original_width != original_width) return original_height < other.original_height;
    if (other.levels != levels) return levels < other.levels;
    if (other.data_bytes != data_bytes) return data_bytes < other.data_bytes;
    return memory_type < other.memory_type;
}

} // namespace vilib
