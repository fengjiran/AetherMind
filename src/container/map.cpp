//
// Created by richard on 11/28/25.
//

#include "container/map.h"

namespace aethermind {

MapImpl<SmallMapImpl>::iterator SmallMapImpl::find_impl(const key_type& key) const {
    const auto* p = static_cast<KVType*>(data_);
    for (size_t i = 0; i < size(); ++i) {
        if (key == p[i].first) {
            return {i, this};
        }
    }

    return end();
}

MapImpl<SmallMapImpl>::value_type& SmallMapImpl::at_impl(const key_type& key) {
    const auto iter = find(key);
    if (iter == end()) {
        AETHERMIND_THROW(KeyError) << "key is not int map.";
    }
    return iter->second;
}

const MapImpl<SmallMapImpl>::value_type& SmallMapImpl::at_impl(const key_type& key) const {
    const auto iter = find(key);
    if (iter == end()) {
        AETHERMIND_THROW(KeyError) << "key is not int map.";
    }
    return iter->second;
}

ObjectPtr<SmallMapImpl> SmallMapImpl::CreateImpl(size_t n) {
    CHECK(n <= kThreshold);
    auto impl = make_array_object<SmallMapImpl, KVType>(n);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(SmallMapImpl);
    impl->size_ = 0;
    impl->slots_ = n;
    return impl;
}

ObjectPtr<SmallMapImpl> SmallMapImpl::CopyFromImpl(const SmallMapImpl* src) {
    const auto* first = static_cast<KVType*>(src->data_);
    const auto* last = first + src->size();
    return CreateFromRangeImpl(first, last);
}

ObjectPtr<Object> SmallMapImpl::InsertMaybeRehash(const KVType& kv, ObjectPtr<Object> old_impl) {
    auto* map = static_cast<SmallMapImpl*>(old_impl.get());//NOLINT
    if (const auto iter = map->find(kv.first); iter != map->end()) {
        iter->second = kv.second;
        return old_impl;
    }

    if (map->size() < map->slots()) {
        auto* p = static_cast<KVType*>(map->data_) + map->size();
        new (p) KVType(kv);
        ++map->size_;
        return old_impl;
    }

    size_t new_cap = std::min(kThreshold, std::max(kIncFactor * map->slots(), kInitSize));
    auto new_impl = CreateImpl(new_cap);
    auto* src = static_cast<KVType*>(map->data_);
    auto* dst = static_cast<KVType*>(new_impl->data_);
    for (size_t i = 0; i < map->size(); ++i) {
        new (dst++) KVType(std::move(*src++));
        ++new_impl->size_;
    }
    new (dst) KVType(kv);
    ++new_impl->size_;
    return new_impl;
}

void SmallMapImpl::erase_impl(const iterator& pos) {
    const auto idx = pos.index();
    if (pos.ptr_ == nullptr || idx >= size()) {
        return;
    }

    auto* p = static_cast<KVType*>(data_);
    p[idx].~KVType();

    auto n = size() - idx - 1;
    auto from = idx + 1;
    auto to = idx;
    for (size_t i = 0; i < n; ++i) {
        p[to] = std::move(p[from]);
        ++to;
        ++from;
    }

    size_ -= 1;
}


SmallMapImpl::~SmallMapImpl() {
    auto* p = static_cast<KVType*>(data_);
    for (size_t i = 0; i < size(); ++i) {
        p[i].~KVType();
    }
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

class DenseMapImpl::ListNode {
public:
    ListNode() : index_(0), obj_(nullptr) {}

    ListNode(size_t index, const DenseMapImpl* p) : index_(index), obj_(p) {}

    NODISCARD size_t index() const {
        return index_;
    }

    NODISCARD const DenseMapImpl* obj() const {
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
    void CreateHead(Entry entry) const {
        GetMeta() = 0x00;
        GetEntry() = std::move(entry);
    }

    // Construct a tail of linked list inplace
    void CreateTail(Entry entry) const {
        GetMeta() = 0x80;
        GetEntry() = std::move(entry);
    }

    // Whether the entry has the next entry on the linked list
    NODISCARD bool HasNext() const {
        return NextProbePosOffset[GetMeta() & 0x7F] != 0;
    }

    // Move to the next entry on the linked list
    bool MoveToNext(std::optional<uint8_t> meta_opt = std::nullopt) {
        uint8_t meta = meta_opt ? meta_opt.value() : GetMeta();
        const auto offset = NextProbePosOffset[meta & 0x7F];
        if (offset == 0) {
            reset();
            return false;
        }

        // The probing will go to the next pos and round back to stay within
        // the correct range of the slots.
        index_ = (index_ + offset) % obj()->slots();
        return true;
    }

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

    NODISCARD std::optional<std::pair<uint8_t, ListNode>> GetNextEmptySlot() const {
        for (uint8_t i = 1; i < kNumOffsetDists; ++i) {
            if (ListNode candidate((index() + NextProbePosOffset[i]) % obj()->slots(), obj());
                candidate.IsEmpty()) {
                return std::make_pair(i, candidate);
            }
        }
        return std::nullopt;
    }

private:
    // Index of entry on the array
    size_t index_;
    // Pointer to the current DenseMapObj
    const DenseMapImpl* obj_;
};

size_t DenseMapImpl::count_impl(const key_type& key) const {
    return !Search(key).IsNone();
}

MapImpl<DenseMapImpl>::iterator DenseMapImpl::find_impl(const key_type& key) const {
    ListNode node = Search(key);
    return node.IsNone() ? end() : iterator(node.index(), this);
}

void DenseMapImpl::erase_impl(const iterator& pos) {
    const auto idx = pos.index();
    if (pos.ptr_ == nullptr || idx >= size()) {
        return;
    }

    ListNode node(idx, this);
    if (node.HasNext()) {
        ListNode prev = node;
        ListNode last = node;
        last.MoveToNext();
        while (last.HasNext()) {
            prev = last;
            last.MoveToNext();
        }

        // needs to first unlink node from the list
        IterListUnlink(node);
        // move data from last to node
        node.GetData() = std::move(last.GetData());
        // Move link chain of iter to last as we stores last node to the new iter loc.
        IterListReplaceNodeBy(last, node);
        last.DestructData();
        last.SetEmpty();
        prev.SetOffset(0);
    } else {// the last node
        if (!node.IsHead()) {
            // cut the link if there is any
            node.FindPrev().SetOffset(0);
        }
        // unlink the node from iterator list
        IterListUnlink(node);
        node.DestructData();
        node.SetEmpty();
    }
    size_ -= 1;
}


DenseMapImpl::Block* DenseMapImpl::GetBlock(size_t block_idx) const {
    return static_cast<Block*>(data_) + block_idx;
}

DenseMapImpl::ListNode DenseMapImpl::IndexFromHash(size_t hash_value) const {
    return {details::FibonacciHash(hash_value, fib_shift_), this};
}

MapImpl<DenseMapImpl>::KVType* DenseMapImpl::DeRefIter(size_t index) const {
    return &ListNode(index, this).GetData();
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

MapImpl<DenseMapImpl>::value_type& DenseMapImpl::At(const key_type& key) const {
    const ListNode iter = Search(key);
    if (iter.IsNone()) {
        AETHERMIND_THROW(KeyError) << "Key not found";
    }

    return iter.GetValue();
}

DenseMapImpl::ListNode DenseMapImpl::Search(const key_type& key) const {
    if (empty()) {
        return {};
    }

    auto node = GetListHead(AnyHash()(key));
    while (!node.IsNone()) {
        if (key == node.GetKey()) {
            return node;
        }
        node.MoveToNext();
    }

    return {};
}

void DenseMapImpl::reset() {
    size_t block_num = ComputeBlockNum(slots());
    auto* p = static_cast<Block*>(data_);
    for (size_t i = 0; i < block_num; ++i) {
        p[i].~Block();
    }

    size_ = 0;
    slots_ = 0;
    fib_shift_ = 63;
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

std::optional<DenseMapImpl::ListNode> DenseMapImpl::TrySpareListHead(ListNode target, const key_type& key) {
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
    ListNode w = target.FindPrev();
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    uint8_t r_meta;

    do {
        auto empty_slot_info = w.GetNextEmptySlot();
        if (!empty_slot_info) {
            return std::nullopt;
        }

        uint8_t offset_idx = empty_slot_info->first;
        ListNode empty = empty_slot_info->second;

        // move `r` to `empty`
        // first move the data over
        empty.CreateTail(Entry(std::move(r.GetData())));
        // then move link list chain of r to empty
        // this needs to happen after NewTail so empty's prev/next get updated
        IterListReplaceNodeBy(r, empty);
        // explicit call destructor to destroy the item in `r`
        r.DestructData();
        // clear the metadata of `r`
        r_meta = r.GetMeta();
        if (is_first) {
            is_first = false;
            r.SetProtected();
        } else {
            r.SetEmpty();
        }
        // link `w` to `empty`, and move forward
        w.SetOffset(offset_idx);
        w = empty;
    } while (r.MoveToNext(r_meta));// move `r` forward as well

    // finally, we have done moving the linked list
    // fill data_ into `target`
    target.CreateHead(Entry(key, value_type(nullptr)));
    size_ += 1;
    return target;
}

std::optional<DenseMapImpl::ListNode> DenseMapImpl::TryInsert(const key_type& key) {
    if (slots() == 0) {
        return std::nullopt;
    }

    // required that `iter` to be the head of a linked list through which we can iterator.
    // `iter` can be:
    // 1) empty;
    // 2) body of an irrelevant list;
    // 3) head of the relevant list.
    ListNode iter = IndexFromHash(AnyHash()(key));

    // Case 1: empty
    if (iter.IsEmpty()) {
        iter.CreateHead(Entry(key, value_type(nullptr)));
        size_ += 1;
        return iter;
    }

    // Case 2: body of an irrelevant list
    if (!iter.IsHead()) {
        // Move the elements around and construct the single-element linked list
        return IsFull() ? std::nullopt : TrySpareListHead(iter, key);
    }

    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    // make sure `iter` is the prev element of `cur`
    ListNode cur = iter;
    do {
        // find equal item, do not insert
        if (key == cur.GetKey()) {
            // we plan to take next, so we need to unlink it from iterator list
            IterListUnlink(cur);
            return cur;
        }
        // make sure `iter` is the previous element of `cur`
        iter = cur;
    } while (cur.MoveToNext());

    // `iter` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
        return std::nullopt;
    }

    // find the next empty slot
    auto empty_slot_info = iter.GetNextEmptySlot();
    if (!empty_slot_info) {
        return std::nullopt;
    }

    uint8_t offset_idx = empty_slot_info->first;
    ListNode empty = empty_slot_info->second;
    empty.CreateTail(Entry(key, value_type(nullptr)));
    // link `iter` to `empty`, and move forward
    iter.SetOffset(offset_idx);
    size_ += 1;
    return empty;
}

ObjectPtr<Object> DenseMapImpl::InsertMaybeRehash(const KVType& kv, ObjectPtr<Object> old_impl) {
    auto* map = static_cast<DenseMapImpl*>(old_impl.get());// NOLINT

    if (auto opt = map->TryInsert(kv.first)) {
        auto node = opt.value();
        node.GetValue() = kv.second;
        map->IterListPushBack(node);
        return old_impl;
    }

    // Otherwise, start rehash
    auto new_impl = CreateImpl(map->fib_shift_ - 1, map->slots() * kIncFactor);

    // need to insert in the same order as the original map
    size_t idx = map->iter_list_head_;
    while (idx != kInvalidIndex) {
        ListNode node(idx, map);
        auto opt = new_impl->TryInsert(node.GetKey());
        opt->GetValue() = std::move(node.GetValue());
        new_impl->IterListPushBack(opt.value());
        idx = node.GetEntry().next;
    }

    auto opt = new_impl->TryInsert(kv.first);
    opt->GetValue() = kv.second;
    new_impl->IterListPushBack(opt.value());

    return new_impl;
}


std::pair<uint32_t, size_t> DenseMapImpl::ComputeSlotNum(size_t cap) {
    uint32_t shift = 64;
    size_t slots = 1;
    size_t c = cap;
    while (c > 0) {
        --shift;
        slots <<= 1;
        c >>= 1;
    }
    CHECK(slots > cap);

    if (slots < 2 * cap) {
        --shift;
        slots <<= 1;
    }

    return {shift, slots};
}

ObjectPtr<DenseMapImpl> DenseMapImpl::CreateImpl(uint32_t fib_shift, size_t slots) {
    CHECK(slots > kThreshold);
    const size_t block_num = ComputeBlockNum(slots);
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
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

ObjectPtr<DenseMapImpl> DenseMapImpl::CreateImpl_(size_t n) {
    CHECK(n >= kThreshold);
    auto [fib_shift, slots] = ComputeSlotNum(n);
    const size_t block_num = ComputeBlockNum(slots);
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
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


ObjectPtr<DenseMapImpl> DenseMapImpl::CopyFromImpl(const DenseMapImpl* src) {
    auto block_num = ComputeBlockNum(src->slots());
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
    impl->size_ = src->size();
    impl->slots_ = src->slots();
    impl->fib_shift_ = src->fib_shift_;
    impl->iter_list_head_ = src->iter_list_head_;
    impl->iter_list_tail_ = src->iter_list_tail_;

    auto* p = static_cast<Block*>(impl->data_);
    for (size_t i = 0; i < block_num; ++i) {
        new (p++) Block(*src->GetBlock(i));
    }

    return impl;
}

const size_t DenseMapImpl::NextProbePosOffset[kNumOffsetDists] = {
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
