//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_MAP_H
#define AETHERMIND_MAP_H

#include "any.h"
#include "container/container_utils.h"
#include "object.h"
#include "object_allocator.h"

namespace aethermind {

class MapImpl : public Object {
public:
    using key_type = Any;
    using value_type = Any;
    using KVType = std::pair<key_type, value_type>;

    static_assert(sizeof(KVType) == 16);

    MapImpl() : data_(nullptr), size_(0), slot_(0) {}

    size_t size() const {
        return size_;
    }

private:
    void* data_;
    size_t size_;
    size_t slot_;
};

template<typename K, typename V>
class Map : public ObjectRef {
public:
private:

};

}// namespace aethermind

#endif//AETHERMIND_MAP_H
