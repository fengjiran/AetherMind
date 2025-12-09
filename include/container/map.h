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

    NODISCARD bool IsSmallMap() const {
        return (slot_ & kSmallMapMask) != 0ull;
    }
};

class SmallMapImpl : public MapImpl {
public:
    using MapImpl::iterator;

    NODISCARD size_t GetSlotNum() const {
        return slot_ & ~kSmallMapMask;
    }

private:
    static constexpr size_t kInitSize = 2;
    static constexpr size_t kMaxSize = 4;

    NODISCARD size_t GetSize() const {
        return size_;
    }

    NODISCARD KVType* GetItemPtr(size_t index) const {
        return static_cast<KVType*>(data_) + index;
    }

    NODISCARD size_t IncIter(size_t index) const {
        return index + 1 < size_ ? index + 1 : size_;
    }

    NODISCARD size_t DecIter(size_t index) const {
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


/*! \brief A specialization of hash map that implements the idea of array-based hash map.
 * Another reference implementation can be found [1].
 *
 * A. Overview
 *
 * DenseMapObj did several improvements over traditional separate chaining hash,
 * in terms of cache locality, memory footprints and data organization.
 *
 * A1. Implicit linked list. For better cache locality, instead of using linked list
 * explicitly for each bucket, we store list data into a single array that spans contiguously
 * in memory, and then carefully design access patterns to make sure most of them fall into
 * a single cache line.
 *
 * A2. 1-byte metadata. There is only 1 byte overhead for each slot in the array to indexing and
 * traversal. This can be divided in 3 parts.
 * 1) Reserved code: (0b11111111)_2 indicates a slot is empty; (0b11111110)_2 indicates protected,
 * which means the slot is empty but not allowed to be written.
 * 2) If not empty or protected, the highest bit is used to indicate whether data in the slot is
 * head of a linked list.
 * 3) The rest 7 bits are used as the "next pointer" (i.e. pointer to the next element). On 64-bit
 * architecture, an ordinary pointer can take up to 8 bytes, which is not acceptable overhead when
 * dealing with 16-byte ObjectRef pairs. Based on a commonly noticed fact that the lists are
 * relatively short (length <= 3) in hash maps, we follow [1]'s idea that only allows the pointer to
 * be one of the 126 possible values, i.e. if the next element of i-th slot is (i + x)-th element,
 * then x must be one of the 126 pre-defined values.
 *
 * A3. Data blocking. We organize the array in the way that every 16 elements forms a data block.
 * The 16-byte metadata of those 16 elements is stored together, followed by the real data, i.e.
 * 16 key-value pairs.
 *
 * B. Implementation details
 *
 * B1. Power-of-2 table size and Fibonacci Hashing. We use power-of-two as table size to avoid
 * modulo for more efficient arithmetics. To make the hash-to-slot mapping distribute more evenly,
 * we use the Fibonacci Hashing [2] trick.
 *
 * B2. Traverse a linked list in the array.
 * 1) List head. Assume Fibonacci Hashing maps a given key to slot i, if metadata at slot i
 * indicates that it is list head, then we found the head; otherwise the list is empty. No probing
 * is done in this procedure. 2) Next element. To find the next element of a non-empty slot i, we
 * look at the last 7 bits of the metadata at slot i. If they are all zeros, then it is the end of
 * the list; otherwise, we know that the next element is (i + candidates[the-last-7-bits]).
 *
 * B3. InsertMaybeReHash an element. Following B2, we first traverse the linked list to see if this
 * element is in the linked list, and if not, we put it at the end by probing the next empty
 * position in one of the 126 candidate positions. If the linked list does not even exist, but the
 * slot for list head has been occupied by another linked list, we should find this intruder another
 * place.
 *
 * B4. Quadratic probing with triangle numbers. In open address hashing, it is provable that probing
 * with triangle numbers can traverse power-of-2-sized table [3]. In our algorithm, we follow the
 * suggestion in [1] that also use triangle numbers for "next pointer" as well as sparing for list
 * head.
 *
 * [1] https://github.com/skarupke/flat_hash_map
 * [2] https://programmingpraxis.com/2018/06/19/fibonacci-hash/
 * [3] https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
 */
class DenseMapImpl : public MapImpl {
public:
    NODISCARD size_t GetSlotNum() const {
        return slot_;
    }

private:
    // The number of elements in a memory block.
    static constexpr int kBlockSize = 16;
    // Max load factor of hash table
    static constexpr double kMaxLoadFactor = 0.99;
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
        uint8_t bytes[kBlockSize + kBlockSize * sizeof(Entry)];

        Block() {// NOLINT
            auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
            for (int i = 0; i < kBlockSize; ++i) {
                bytes[i] = kEmptySlot;
                new (data++) Entry;
            }
        }

        Block(const Block& other) {// NOLINT
            auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
            for (int i = 0; i < kBlockSize; ++i) {
                bytes[i] = other.bytes[i];
                new (data++) Entry(reinterpret_cast<const Entry*>(other.bytes + kBlockSize)[i]);
            }
        }

        ~Block() {
            auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
            for (int i = 0; i < kBlockSize; ++i) {
                bytes[i] = kEmptySlot;
                data->~Entry();
            }
        }
    };

    class ListNode {
    public:
        ListNode() : index_(0), block_(nullptr) {}

        ListNode(size_t index, const DenseMapImpl* p)
            : index_(index), block_(p->GetBlock(index / kBlockSize)) {}

        NODISCARD size_t index() const {
            return index_;
        }

        // Get metadata of an entry
        NODISCARD uint8_t& GetMetadata() const {
            return *(block_->bytes + index_ % kBlockSize);
        }

        // Get an entry ref
        NODISCARD Entry& GetEntry() const {
            auto* p = reinterpret_cast<Entry*>(block_->bytes + kBlockSize);
            return *(p + index_ % kBlockSize);
        }

        // Get KV
        NODISCARD KVType& GetData() const {
            return GetEntry().data;
        }

        NODISCARD key_type& GetKey() const {
            return GetData().first;
        }

        NODISCARD value_type& GetValue() const {
            return GetData().second;
        }

        NODISCARD bool IsNone() const {
            return block_ == nullptr;
        }

        NODISCARD bool IsEmpty() const {
            return GetMetadata() == kEmptySlot;
        }

        NODISCARD bool IsProtected() const {
            return GetMetadata() == kProtectedSlot;
        }

        NODISCARD bool IsHead() const {
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
            CHECK(jump < kNumJumpDists);
            (GetMetadata() &= 0x80) |= jump;
        }

        // Destroy the item in the entry.
        void DestructData() const {
            GetKey().~key_type();
            GetValue().~value_type();
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
        NODISCARD bool HasNext() const {
            return NextProbePosOffset[GetMetadata() & 0x7F] != 0;
        }

        // Move to the next entry on the linked list
        bool MoveToNext(const DenseMapImpl* p, uint8_t meta) {
            auto offset = NextProbePosOffset[meta & 0x7F];
            if (offset == 0) {
                index_ = 0;
                block_ = nullptr;
                return false;
            }

            // The probing will go to the next pos and round back to stay within
            // the correct range of the slots.
            index_ = (index_ + offset) % p->GetSlotNum();
            block_ = p->GetBlock(index_ / kBlockSize);
            return true;
        }

        bool MoveToNext(const DenseMapImpl* p) {
            return MoveToNext(p, GetMetadata());
        }

        // Get the prev entry on the linked list
        ListNode FindPrev(const DenseMapImpl* p) const {
            // start from the head of the linked list, which must exist
            auto cur = p->IndexFromHash(AnyHash()(GetKey()));
            auto prev = cur;

            cur.MoveToNext(p);
            while (index_ != cur.index_) {
                prev = cur;
                cur.MoveToNext(p);
            }

            return prev;
        }

        bool GetNextEmpty(const DenseMapImpl* p, uint8_t* offset_idx, ListNode* res) const {
            for (uint8_t i = 1; i < kNumJumpDists; ++i) {
                if (ListNode candidate((index_ + NextProbePosOffset[i]) % p->GetSlotNum(), p);
                    candidate.IsEmpty()) {
                    *offset_idx = i;
                    *res = candidate;
                    return true;
                }
            }
            return false;
        }

    private:
        // Index of entry on the array
        size_t index_;
        // Pointer to the actual block
        Block* block_;
    };

    NODISCARD Block* GetBlock(size_t block_index) const {
        return static_cast<Block*>(data_) + block_index;
    }

    NODISCARD ListNode IndexFromHash(size_t hash_value) const {
        return {details::FibonacciHash(hash_value, fib_shift_), this};
    }

    static size_t ComputeBlockNum(size_t slot_num) {
        return (slot_num + kBlockSize - 1) / kBlockSize;
    }

    // Construct a ListNode from hash code if the position is head of list
    NODISCARD ListNode GetListHead(size_t hash_value) const {
        const auto node = IndexFromHash(hash_value);
        return node.IsHead() ? node : ListNode();
    }

    // Whether the hash table is full.
    NODISCARD bool IsFull() const {
        return size() + 1 > static_cast<size_t>(static_cast<double>(GetSlotNum()) * kMaxLoadFactor);
    }

    NODISCARD size_t IncIter(size_t index) const {
        // keep at the end of iterator
        if (index == kInvalidIndex) {
            return index;
        }

        return ListNode(index, this).GetEntry().next;
    }

    NODISCARD size_t DecIter(size_t index) const {
        // this is the end iterator, we need to return tail.
        if (index == kInvalidIndex) {
            return iter_list_tail_;
        }

        return ListNode(index, this).GetEntry().prev;
    }

    NODISCARD ListNode Search(const key_type& key) const {
        if (size_ == 0) {
            return {};
        }

        ListNode iter = GetListHead(AnyHash()(key));
        while (!iter.IsNone()) {
            if (iter.GetKey() == key) {
                return iter;
            }
            iter.MoveToNext(this);
        }
        return {};
    }

    // Insert the entry into tail of iterator list.
    // This function does not change data content of the node.
    void IterListPushBack(ListNode node) {
        node.GetEntry().prev = iter_list_tail_;
        node.GetEntry().next = kInvalidIndex;

        if (iter_list_tail_ != kInvalidIndex) {
            ListNode(iter_list_tail_, this).GetEntry().next = node.index();
        }

        if (iter_list_head_ == kInvalidIndex) {
            iter_list_head_ = node.index();
        }

        iter_list_tail_ = node.index();
    }

    // Unlink the entry from iterator list.
    // This function is usually used before deletion,
    // and it does not change data content of the node.
    void IterListUnlink(ListNode node) {
        if (node.GetEntry().prev == kInvalidIndex) {
            iter_list_head_ = node.GetEntry().next;
        } else {
            ListNode prev_node(node.GetEntry().prev, this);
            prev_node.GetEntry().next = node.GetEntry().next;
        }

        if (node.GetEntry().next == kInvalidIndex) {
            iter_list_tail_ = node.GetEntry().prev;
        } else {
            ListNode next_node(node.GetEntry().next, this);
            next_node.GetEntry().prev = node.GetEntry().prev;
        }
    }

    /*!
   * \brief Replace node src by dst in the iter list
   * \param src The source node
   * \param dst The destination node, must be empty
   * \note This function does not change data content of the nodes,
   *       which needs to be updated by the caller.
   */
    void IterListReplaceNodeBy(ListNode src, ListNode dst) {
        dst.GetEntry().prev = src.GetEntry().prev;
        dst.GetEntry().next = src.GetEntry().next;

        if (dst.GetEntry().prev == kInvalidIndex) {
            iter_list_head_ = dst.index();
        } else {
            ListNode prev_node(dst.GetEntry().prev, this);
            prev_node.GetEntry().next = dst.index();
        }

        if (dst.GetEntry().next == kInvalidIndex) {
            iter_list_tail_ = dst.index();
        } else {
            ListNode next_node(dst.GetEntry().next, this);
            next_node.GetEntry().prev = dst.index();
        }
    }

    /*!
   * \brief Spare an entry to be the head of a linked list.
   * As described in B3, during insertion, it is possible that the entire linked list does not
   * exist, but the slot of its head has been occupied by other linked lists. In this case, we need
   * to spare the slot by moving away the elements to another valid empty one to make insertion
   * possible.
   * \param target The given entry to be spared
   * \param key The indexing key
   * \param result The linked-list entry constructed as the head
   * \return A boolean, if actual insertion happens
   */
    bool TrySpareListHead(ListNode target, const key_type& key, ListNode* result);

    /*!
   * \brief Try to insert a key, or do nothing if already exists
   * \param key The indexing key
   * \param result The linked-list entry found or just constructed
   * \return A boolean, indicating if actual insertion happens
   */
    bool TryInsert(const key_type& key, ListNode* result);

    static ObjectPtr<DenseMapImpl> Create(uint32_t fib_shift, size_t slots);

    static ObjectPtr<DenseMapImpl> CopyFrom(const DenseMapImpl* src);

    // Calculate the power-of-2 table size given the lower-bound of required capacity.
    static void ComputeTableSize(size_t cap, uint32_t* fib_shift, size_t* n_slots);
};

template<typename K, typename V>
class Map : public ObjectRef {
public:
private:
    ObjectPtr<MapImpl> impl_;
};

}// namespace aethermind

#endif//AETHERMIND_MAP_H
