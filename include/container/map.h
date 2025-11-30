//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_MAP_H
#define AETHERMIND_MAP_H

#include "any.h"
#include "container/container_utils.h"
#include "map.h"
#include "object.h"
#include "object_allocator.h"

namespace aethermind {

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

    size_t GetSize() const {
        return size_;
    }

    KVType* GetItemPtr(size_t index) const {
        return static_cast<KVType*>(data_) + index;
    }

    size_t IncIter(size_t index) const {
        return index + 1 < size_ ? index + 1 : size_;
    }

    size_t DecIter(size_t index) const {
        return index > 0 ? index - 1 : size_;
    }

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

    friend class MapImpl;
    friend class DenseMapImpl;
};

class DenseMapImpl : public MapImpl {
public:
    size_t GetSlotNum() const {
        return slot_;
    }

private:
    // The number of elements in a memory block.
    static constexpr int kBlockCap = 16;
    // 0b11111111 representation of the metadata of an empty slot.
    static constexpr uint8_t kEmptySlot = 0xFF;
    // 0b11111110 representation of the metadata of a protected slot.
    static constexpr uint8_t kProtectedSlot = 0xFE;
    // Number of probing choices available
    static constexpr int kNumJumpDists = 126;
    // Index indicator to indicate an invalid index.
    static constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();

    static const size_t NextProbePosOffset[kNumJumpDists];

    // fib shift in Fibonacci hash
    uint32_t fib_shift_;
    // The head of iterator list
    size_t iter_list_head_ = kInvalidIndex;
    // The tail of iterator list
    size_t iter_list_tail_ = kInvalidIndex;

    struct Entry {
        KVType data{};
        size_t prev = kInvalidIndex;
        size_t next = kInvalidIndex;

        Entry() = default;
        explicit Entry(KVType&& data) : data(std::move(data)) {}
        explicit Entry(key_type key, value_type value) : data(std::move(key), std::move(value)) {}
    };

    struct Block {
        uint8_t bytes[kBlockCap + kBlockCap * sizeof(Entry)];

        Block() {
            auto* p = reinterpret_cast<Entry*>(bytes + kBlockCap);
            for (int i = 0; i < kBlockCap; ++i) {
                bytes[i] = kEmptySlot;
                new (p++) Entry();
            }
        }

        Block(const Block& other) {
            auto* p = reinterpret_cast<Entry*>(bytes + kBlockCap);
            for (int i = 0; i < kBlockCap; ++i) {
                bytes[i] = other.bytes[i];
                new (p++) Entry(reinterpret_cast<const Entry*>(other.bytes + kBlockCap)[i]);
            }
        }

        ~Block() {
            auto* p = reinterpret_cast<Entry*>(bytes + kBlockCap);
            for (int i = 0; i < kBlockCap; ++i) {
                bytes[i] = kEmptySlot;
                p->~Entry();
            }
        }
    };

    class ListNode {
    public:
        ListNode() : index(0), block(nullptr) {}

        ListNode(size_t index, const DenseMapImpl* p)
            : index(index), block(p->GetBlock(index / kBlockCap)) {}

        // Get metadata of an entry
        uint8_t& GetMetadata() const {
            return *(block->bytes + index % kBlockCap);
        }

        // Get an entry
        Entry& GetEntry() const {
            auto* p = reinterpret_cast<Entry*>(block->bytes + kBlockCap);
            return *(p + index % kBlockCap);
        }

        KVType& GetData() const {
            return GetEntry().data;
        }

        key_type& GetKey() const {
            return GetData().first;
        }

        value_type& GetValue() const {
            return GetData().second;
        }

        bool IsNone() const {
            return block == nullptr;
        }

        bool IsEmpty() const {
            return GetMetadata() == kEmptySlot;
        }

        bool IsProtected() const {
            return GetMetadata() == kProtectedSlot;
        }

        bool IsHead() const {
            return (GetMetadata() & 0x80) == 0x00;
        }

        void SetEmpty() const {
            GetMetadata() = kEmptySlot;
        }

        void SetProtected() const {
            GetMetadata() = kProtectedSlot;
        }

        // Set the entry's jump to its next entry.
        void SetJump(uint8_t jump) const {
            (GetMetadata() &= 0x80) |= jump;
        }

        // Destroy the item in the entry.
        void DestructData() const {
            GetKey().~Any();
            GetValue().~Any();
        }

        // Construct a head of linked list inplace.
        void NewHead(Entry entry) const {
            GetMetadata() = 0x00;
            GetEntry() = std::move(entry);
        }

        // Construct a tail of linked list inplace
        void NewTail(Entry entry) const {
            GetMetadata() = 0x80;
            GetEntry() = std::move(entry);
        }

        // Whether the entry has the next entry on the linked list
        bool HasNext() const {
            return NextProbePosOffset[GetMetadata() & 0x7F] != 0;
        }

        // Move the entry to the next entry on the linked list
        bool MoveToNext(const DenseMapImpl* p, uint8_t meta) {
            auto offset = NextProbePosOffset[meta & 0x7F];
            if (offset == 0) {
                index = 0;
                block = nullptr;
                return false;
            }

            // The probing will go to the next pos and round back to stay within
            // the correct range of the slots.
            index = (index + offset) % p->GetSlotNum();
            block = p->GetBlock(index / kBlockCap);
            return true;
        }

        bool MoveToNext(const DenseMapImpl* p) {
            return MoveToNext(p, GetMetadata());
        }

        // Get the prev entry on the linked list
        ListNode FindPrev(const DenseMapImpl* p) const {
            // start from the head of the linked list, which must exist
            auto next = p->IndexFromHash(AnyHash()(GetKey()));
            auto prev = next;

            next.MoveToNext(p);
            while (index != next.index) {
                prev = next;
                next.MoveToNext(p);
            }

            return prev;
        }

        bool GetNextEmpty(const DenseMapImpl* p, uint8_t* jump, ListNode* res) const {
            for (uint8_t i = 1; i < kNumJumpDists; ++i) {
                ListNode candidate((index + NextProbePosOffset[i]) % p->GetSlotNum(), p);
                if (candidate.IsEmpty()) {
                    *jump = i;
                    *res = candidate;
                    return true;
                }
            }
            return false;
        }

    private:
        size_t index;
        Block* block;
    };

    Block* GetBlock(size_t block_index) const {
        return static_cast<Block*>(data_) + block_index;
    }

    ListNode IndexFromHash(size_t hash_value) const {
        return ListNode(details::FibonacciHash(hash_value, fib_shift_), this);
    }

    static size_t ComputeBlockNum(size_t slot_num) {
        return (slot_num + kBlockCap - 1) / kBlockCap;
    }

    static ObjectPtr<DenseMapImpl> Create(uint32_t fib_shift, size_t slots);

    static ObjectPtr<DenseMapImpl> CopyFrom(const DenseMapImpl* src);
};

template<typename K, typename V>
class Map : public ObjectRef {
public:
private:
    ObjectPtr<MapImpl> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_MAP_H
