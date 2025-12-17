//
// Created by richard on 11/28/25.
//

#include "container/map.h"

namespace aethermind {

class MapImpl::iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = KVType;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = int64_t;

    iterator() : index_(0), ptr_(nullptr) {}

    pointer operator->() const {
        if (ptr_->IsSmallMap()) {
            const auto* p = dynamic_cast<const SmallMapImpl*>(ptr_);
            return p->DeRefIter(index_);
        }
        const auto* p = dynamic_cast<const DenseMapImpl*>(ptr_);
        return p->DeRefIter(index_);
    }

    bool operator==(const iterator& other) const {
        return index_ == other.index_ && ptr_ == other.ptr_;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

protected:
    iterator(size_t index, const MapImpl* ptr) : index_(index), ptr_(ptr) {}

    size_t index_;      // The position in the array.
    const MapImpl* ptr_;// The container it pointer to.

    friend class SmallMapImpl;
    friend class DenseMapImpl;
};

ObjectPtr<SmallMapObj> SmallMapObj::CreateSmallMap(size_t n) {
    CHECK(n <= kMaxSize);
    auto impl = make_array_object<SmallMapObj, KVType>(n);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(SmallMapObj);
    impl->size_ = 0;
    impl->slots_ = n & ~kSmallMapMask | kSmallMapMask;
    return impl;
}

struct DenseMapObj::Entry {
    KVType data{};
    size_t prev = kInvalidIndex;
    size_t next = kInvalidIndex;

    Entry() = default;
    explicit Entry(KVType&& data) : data(std::move(data)) {}
    explicit Entry(key_type key, value_type value) : data(std::move(key), std::move(value)) {}
};

struct DenseMapObj::Block {
    uint8_t bytes[kBlockSize + kBlockSize * sizeof(Entry)];

    Block() {// NOLINT
        auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (int i = 0; i < kBlockSize; ++i) {
            bytes[i] = kEmptySlot;
            new (data++) Entry;
        }
    }

    Block(const Block& other) {// NOLINT
        const auto* src = reinterpret_cast<const Entry*>(other.bytes + kBlockSize);
        auto* dst = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (int i = 0; i < kBlockSize; ++i) {
            bytes[i] = other.bytes[i];
            new (dst++) Entry(src[i]);
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

class DenseMapObj::ListNode {
public:
    ListNode() : index_(0), obj_(nullptr) {}

    ListNode(size_t index, const DenseMapObj* p) : index_(index), obj_(p) {}

    NODISCARD size_t index() const {
        return index_;
    }

    NODISCARD const DenseMapObj* obj() const {
        return obj_;
    }

    void reset() {
        index_ = 0;
        obj_ = nullptr;
    }

    NODISCARD Block* GetBlock() const {
        return obj()->GetBlock(index() / kBlockSize);
    }

    // Get metadata of an entry
    NODISCARD uint8_t& GetMeta() const {
        return GetBlock()->bytes[index() % kBlockSize];
    }

    // Get an entry ref
    NODISCARD Entry& GetEntry() const {
        auto* p = reinterpret_cast<Entry*>(GetBlock()->bytes + kBlockSize);
        return p[index() % kBlockSize];
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
        return obj() == nullptr;
    }

    NODISCARD bool IsEmpty() const {
        return GetMeta() == kEmptySlot;
    }

    NODISCARD bool IsProtected() const {
        return GetMeta() == kProtectedSlot;
    }

    NODISCARD bool IsHead() const {
        return (GetMeta() & 0x80) == 0x00;
    }

    void SetEmpty() const {
        GetMeta() = kEmptySlot;
    }

    void SetProtected() const {
        GetMeta() = kProtectedSlot;
    }

    // Set the entry's offset to its next entry.
    void SetOffset(uint8_t offset_idx) const {
        CHECK(offset_idx < kNumOffsetDists);
        (GetMeta() &= 0x80) |= offset_idx;
    }

    // Destroy the item in the entry.
    void DestructData() const {
        // GetKey().~key_type();
        // GetValue().~value_type();
        GetEntry().~Entry();
    }

    // Construct a head of linked list inplace.
    void NewHead(Entry entry) const {
        GetMeta() = 0x00;
        GetEntry() = std::move(entry);
    }

    // Construct a tail of linked list inplace
    void NewTail(Entry entry) const {
        GetMeta() = 0x80;
        GetEntry() = std::move(entry);
    }

    // Whether the entry has the next entry on the linked list
    NODISCARD bool HasNext() const {
        return NextProbePosOffset[GetMeta() & 0x7F] != 0;
    }

    // Move to the next entry on the linked list
    bool MoveToNext() {
        const auto offset = NextProbePosOffset[GetMeta() & 0x7F];
        if (offset == 0) {
            reset();
            return false;
        }

        // The probing will go to the next pos and round back to stay within
        // the correct range of the slots.
        index_ = (index_ + offset) % obj()->slots();
        return true;
    }

    // // Move to the next entry on the linked list
    // bool MoveToNext(const DenseMapObj* p, uint8_t meta) {
    //     const auto offset = NextProbePosOffset[meta & 0x7F];
    //     if (offset == 0) {
    //         index_ = 0;
    //         block_ = nullptr;
    //         return false;
    //     }
    //
    //     // The probing will go to the next pos and round back to stay within
    //     // the correct range of the slots.
    //     index_ = (index_ + offset) % p->slots();
    //     block_ = p->GetBlock(index_ / kBlockSize);
    //     return true;
    // }
    //
    // bool MoveToNext(const DenseMapObj* p) {
    //     return MoveToNext(p, GetMeta());
    // }

    // Get the prev entry on the linked list
    NODISCARD ListNode FindPrev() const {
        // start from the head of the linked list, which must exist
        auto cur = obj()->IndexFromHash(AnyHash()(GetKey()));
        auto prev = cur;

        cur.MoveToNext();
        while (index() != cur.index()) {
            prev = cur;
            cur.MoveToNext();
        }

        return prev;
    }

    bool GetNextEmpty(uint8_t* offset_idx, ListNode* res) const {
        for (uint8_t i = 1; i < kNumOffsetDists; ++i) {
            if (ListNode candidate((index() + NextProbePosOffset[i]) % obj()->slots(), obj());
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
    // Block* block_;
    // Pointer to the current DenseMapObj
    const DenseMapObj* obj_;
};

DenseMapObj::Block* DenseMapObj::GetBlock(size_t block_idx) const {
    return static_cast<Block*>(data_) + block_idx;
}

DenseMapObj::ListNode DenseMapObj::IndexFromHash(size_t hash_value) const {
    return {details::FibonacciHash(hash_value, fib_shift_), this};
}

MapObj<DenseMapObj>::KVType* DenseMapObj::DeRefIter(size_t index) const {
    return &ListNode(index, this).GetData();
}

size_t DenseMapObj::IncIter(size_t index) const {
    // keep at the end of iterator
    if (index == kInvalidIndex) {
        return index;
    }

    return ListNode(index, this).GetEntry().next;
}

size_t DenseMapObj::DecIter(size_t index) const {
    // this is the end iterator, we need to return tail.
    if (index == kInvalidIndex) {
        return iter_list_tail_;
    }

    return ListNode(index, this).GetEntry().prev;
}


ObjectPtr<DenseMapObj> DenseMapObj::CreateDenseMap(uint32_t fib_shift, size_t slots) {
    CHECK(slots > SmallMapObj::kMaxSize);
    CHECK((slots & kSmallMapMask) == 0ull);
    const size_t block_num = ComputeBlockNum(slots);
    auto impl = make_array_object<DenseMapObj, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapObj);
    impl->size_ = 0;
    impl->slots_ = slots;
    impl->fib_shift_ = fib_shift;
    impl->iter_list_head_ = kInvalidIndex;
    impl->iter_list_tail_ = kInvalidIndex;

    auto* p = static_cast<Block*>(impl->data_);
    for (size_t i = 0; i < block_num; ++i) {
        new (p++) Block;
    }
    return impl;
}


ObjectPtr<SmallMapImpl> SmallMapImpl::Create(size_t n) {
    auto impl = make_array_object<SmallMapImpl, KVType>(n);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(SmallMapImpl);
    impl->size_ = 0;
    impl->slots_ = n & ~kSmallMapMask | kSmallMapMask;
    return impl;
}

ObjectPtr<SmallMapImpl> SmallMapImpl::CopyFrom(const SmallMapImpl* src) {
    auto* first = static_cast<KVType*>(src->data_);
    auto* last = first + src->size_;
    return CreateFromRange(src->size_, first, last);
}

struct DenseMapImpl::Entry {
    KVType data{};
    size_t prev = kInvalidIndex;
    size_t next = kInvalidIndex;

    Entry() = default;
    explicit Entry(KVType&& data) : data(std::move(data)) {}
    explicit Entry(key_type key, value_type value) : data(std::move(key), std::move(value)) {}
};

struct DenseMapImpl::Block {
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

class DenseMapImpl::ListNode {
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

DenseMapImpl::Block* DenseMapImpl::GetBlock(size_t block_index) const {
    return static_cast<Block*>(data_) + block_index;
}

DenseMapImpl::ListNode DenseMapImpl::IndexFromHash(size_t hash_value) const {
    return {details::FibonacciHash(hash_value, fib_shift_), this};
}

DenseMapImpl::ListNode DenseMapImpl::GetListHead(size_t hash_value) const {
    const auto node = IndexFromHash(hash_value);
    return node.IsHead() ? node : ListNode();
}

size_t DenseMapImpl::IncIter(size_t index) const {
    // keep at the end of iterator
    if (index == kInvalidIndex) {
        return index;
    }

    return ListNode(index, this).GetEntry().next;
}

size_t DenseMapImpl::DecIter(size_t index) const {
    // this is the end iterator, we need to return tail.
    if (index == kInvalidIndex) {
        return iter_list_tail_;
    }

    return ListNode(index, this).GetEntry().prev;
}

MapImpl::KVType* DenseMapImpl::DeRefIter(size_t index) const {
    return &ListNode(index, this).GetData();
}

void DenseMapImpl::reset() {
    size_t block_num = ComputeBlockNum(this->GetSlotNum());
    auto* p = static_cast<Block*>(data_);
    for (size_t i = 0; i < block_num; ++i) {
        p->~Block();
        ++p;
    }

    size_ = 0;
    slots_ = 0;
    fib_shift_ = 63;
}

DenseMapImpl::~DenseMapImpl() {
    reset();
}

ObjectPtr<DenseMapImpl> DenseMapImpl::Create(uint32_t fib_shift, size_t slot_num) {
    CHECK(slot_num > SmallMapImpl::kMaxSize);
    CHECK((slot_num & kSmallMapMask) == 0ull);
    size_t block_num = ComputeBlockNum(slot_num);
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
    impl->size_ = 0;
    impl->slots_ = slot_num;
    impl->fib_shift_ = fib_shift;
    impl->iter_list_head_ = kInvalidIndex;
    impl->iter_list_tail_ = kInvalidIndex;

    auto* p = static_cast<Block*>(impl->data_);
    for (size_t i = 0; i < block_num; ++i) {
        new (p++) Block;
    }
    return impl;
}

ObjectPtr<DenseMapImpl> DenseMapImpl::CopyFrom(const DenseMapImpl* src) {
    CHECK((src->GetSlotNum() & kSmallMapMask) == 0ull);
    auto block_num = ComputeBlockNum(src->GetSlotNum());
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
    impl->size_ = src->size();
    impl->slots_ = src->GetSlotNum();
    impl->fib_shift_ = src->fib_shift_;
    impl->iter_list_head_ = src->iter_list_head_;
    impl->iter_list_tail_ = src->iter_list_tail_;

    auto* p = static_cast<Block*>(impl->data_);
    for (size_t i = 0; i < block_num; ++i) {
        new (p++) Block(*src->GetBlock(i));
    }

    return impl;
}

void DenseMapImpl::ComputeTableSize(size_t cap, uint32_t* fib_shift, size_t* n_slots) {
    uint32_t shift = 64;
    size_t slots = 1;
    size_t c = cap;
    while (c > 0) {
        shift -= 1;
        slots <<= 1;
        c >>= 1;
    }
    CHECK(slots > cap);

    if (slots < 2 * cap) {
        *fib_shift = shift - 1;
        *n_slots = slots << 1;
    } else {
        *fib_shift = shift;
        *n_slots = slots;
    }
}

DenseMapImpl::ListNode DenseMapImpl::Search(const key_type& key) const {
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

size_t DenseMapImpl::count(const key_type& key) const {
    return !Search(key).IsNone();
}

MapImpl::value_type& DenseMapImpl::At(const key_type& key) const {
    const ListNode iter = Search(key);
    if (iter.IsNone()) {
        AETHERMIND_THROW(KeyError) << "Key not found";
    }

    return iter.GetValue();
}

void DenseMapImpl::IterListPushBack(ListNode node) {
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

void DenseMapImpl::IterListUnlink(ListNode node) {
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

void DenseMapImpl::IterListReplaceNodeBy(ListNode src, ListNode dst) {
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


bool DenseMapImpl::TrySpareListHead(ListNode target, const key_type& key, ListNode* result) {
    // `target` is not the head of the linked list
    // move the original item of `target` (if any)
    // and construct new item on the position `target`
    // To make `target` empty, we
    // 1) find `w` the previous element of `target` in the linked list
    // 2) copy the linked list starting from `r = target`
    // 3) paste them after `w`

    // read from the linked list after `r`
    ListNode r = target;
    // write to the tail of `w`
    ListNode w = target.FindPrev(this);
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    uint8_t r_metadata;
    uint8_t offset_idx;
    ListNode empty;

    do {
        if (!w.GetNextEmpty(this, &offset_idx, &empty)) {
            return false;
        }

        // move `r` to `empty`
        // first move the data over
        empty.NewTail(Entry(std::move(r.GetData())));
        // then move link list chain of r to empty
        // this needs to happen after NewTail so empty's prev/next get updated
        IterListReplaceNodeBy(r, empty);
        // explicit call destructor to destroy the item in `r`
        r.DestructData();
        // clear the metadata of `r`
        r_metadata = r.GetMetadata();
        if (is_first) {
            is_first = false;
            r.SetProtected();
        } else {
            r.SetEmpty();
        }
        // link `w` to `empty`, and move forward
        w.SetJump(offset_idx);
        w = empty;
    } while (r.MoveToNext(this, r_metadata));// move `r` forward as well

    // finally we have done moving the linked list
    // fill data_ into `target`
    target.NewHead(Entry(key, value_type(nullptr)));
    size_ += 1;
    *result = target;
    return true;
}

bool DenseMapImpl::TryInsert(const key_type& key, ListNode* result) {
    if (slots_ == 0) {
        return false;
    }

    // required that `iter` to be the head of a linked list through which we can iterator
    ListNode iter = IndexFromHash(AnyHash()(key));
    // `iter` can be: 1) empty; 2) body of an irrelevant list; 3) head of the relevant list

    // Case 1: empty
    if (iter.IsEmpty()) {
        iter.NewHead(Entry(key, value_type(nullptr)));
        size_ += 1;
        *result = iter;
        return true;
    }

    // Case 2: body of an irrelevant list
    if (!iter.IsHead()) {
        // we move the elements around and construct the single-element linked list
        return IsFull() ? false : TrySpareListHead(iter, key, result);
    }

    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    // make sure `iter` is the previous element of `cur`
    ListNode cur = iter;
    do {
        // find equal item, do not insert
        if (key == cur.GetKey()) {
            // we plan to take next, so we need to unlink it from iterator list
            IterListUnlink(cur);
            *result = cur;
            return true;
        }
        // make sure `iter` is the previous element of `cur`
        iter = cur;
    } while (cur.MoveToNext(this));

    // `iter` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
        return false;
    }

    // find the next empty slot
    uint8_t offset_idx;
    if (!iter.GetNextEmpty(this, &offset_idx, result)) {
        return false;
    }

    result->NewTail(Entry(key, value_type(nullptr)));
    // link `iter` to `empty`, and move forward
    iter.SetJump(offset_idx);
    size_ += 1;
    return true;
}


const size_t DenseMapObj::NextProbePosOffset[kNumOffsetDists] = {
        /* clang-format off */
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      // Quadratic probing with triangle numbers. See also:
      // 1) https://en.wikipedia.org/wiki/Quadratic_probing
      // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
      // 3) https://github.com/skarupke/flat_hash_map
      21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
      136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
      351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
      666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
      1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
      1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
      2211, 2278, 2346, 2415, 2485, 2556, 2628,
      // larger triangle numbers
      8515, 19110, 42778, 96141, 216153,
      486591, 1092981, 2458653, 5532801, 12442566,
      27993903, 62983476, 141717030, 318844378, 717352503,
      1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
      93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
      5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
      309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
      17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251,
      457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435,
};


const size_t DenseMapImpl::NextProbePosOffset[kNumJumpDists] = {
        /* clang-format off */
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      // Quadratic probing with triangle numbers. See also:
      // 1) https://en.wikipedia.org/wiki/Quadratic_probing
      // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
      // 3) https://github.com/skarupke/flat_hash_map
      21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
      136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
      351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
      666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
      1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
      1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
      2211, 2278, 2346, 2415, 2485, 2556, 2628,
      // larger triangle numbers
      8515, 19110, 42778, 96141, 216153,
      486591, 1092981, 2458653, 5532801, 12442566,
      27993903, 62983476, 141717030, 318844378, 717352503,
      1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
      93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
      5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
      309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
      17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251,
      457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435,
};

}// namespace aethermind
