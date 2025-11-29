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

struct Item {
    std::pair<Any, Any> data;
    size_t prev = std::numeric_limits<size_t>::max();
    size_t next = std::numeric_limits<size_t>::max();
};

struct Block {
    uint8_t bytes[16 + 16 * sizeof(Item)];

    Block() {
        auto* p = reinterpret_cast<Item*>(bytes + 16);
        for (int i = 0; i < 16; ++i) {
            bytes[i] = 0xFF;
            new (p++) Item();
        }
    }

    ~Block() {
        auto* p = reinterpret_cast<Item*>(bytes + 16);
        for (int i = 0; i < 16; ++i) {
            p->~Item();
        }
    }
};

class MapImpl : public Object {
public:
    using key_type = Any;
    using value_type = Any;
    using KVType = std::pair<key_type, value_type>;

    static_assert(sizeof(KVType) == 16);

    class iterator;

    MapImpl() : data_(nullptr), size_(0), slot_(0) {}

    NODISCARD size_t size() const {
        return size_;
    }

protected:
    void* data_;
    size_t size_;
    size_t slot_;

    // Small map mask, the most significant bit is used to indicate the small map layout.
    static constexpr size_t kSmallMapMask = static_cast<size_t>(1) << 63;

    bool IsSmallMap() const {
        return (slot_ & kSmallMapMask) != 0ull;
    }
};

class SmallMapImpl : public MapImpl {
public:
    using MapImpl::iterator;

    size_t GetSlotNum() const {
        return slot_ & ~kSmallMapMask;
    }

private:
    static constexpr size_t kInitSize = 2;
    static constexpr size_t kMaxSize = 4;

    static ObjectPtr<SmallMapImpl> Create(size_t n = kInitSize);

    template<typename Iter>
    static ObjectPtr<SmallMapImpl> CreateFromRange(size_t n, Iter first, Iter last) {
        ObjectPtr<SmallMapImpl> impl = Create(n);
        auto* ptr = static_cast<KVType*>(impl->data_);
        while (first != last) {
            new (ptr++) KVType(*first++);
        }
        return impl;
    }

    static ObjectPtr<SmallMapImpl> CopyFrom(const SmallMapImpl* src);
};

class DenseMapImpl : public MapImpl {
};

template<typename K, typename V>
class Map : public ObjectRef {
public:
private:
    ObjectPtr<MapImpl> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_MAP_H
