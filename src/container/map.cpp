//
// Created by richard on 11/28/25.
//

#include "container/map.h"

namespace aethermind {

MapImpl<SmallMapImpl>::iterator SmallMapImpl::find_impl(const key_type& key) {
    const auto* p = static_cast<KVType*>(data());
    for (size_t i = 0; i < size(); ++i) {
        if (key == p[i].first) {
            return {i, this};
        }
    }

    return end();
}

MapImpl<SmallMapImpl>::const_iterator SmallMapImpl::find_impl(const key_type& key) const {
    return const_cast<SmallMapImpl*>(this)->find_impl(key);
}

MapImpl<SmallMapImpl>::mapped_type& SmallMapImpl::at_impl(const key_type& key) {
    const auto iter = find(key);
    if (iter == end()) {
        AETHERMIND_THROW(KeyError) << "key is not exist.";
    }
    return iter->second;
}

const MapImpl<SmallMapImpl>::mapped_type& SmallMapImpl::at_impl(const key_type& key) const {
    return const_cast<SmallMapImpl*>(this)->at_impl(key);
}

void SmallMapImpl::reset() {
    auto* p = static_cast<KVType*>(data());
    for (size_t i = 0; i < size(); ++i) {
        p[i].~KVType();
    }

    size_ = 0;
    slots_ = 0;
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
    const auto* first = static_cast<KVType*>(src->data());
    auto impl = Create(src->slots());
    auto* p = static_cast<SmallMapImpl*>(impl.get());//NOLINT
    auto* dst = static_cast<KVType*>(p->data());
    for (size_t i = 0; i < src->size(); ++i) {
        new (dst + i) KVType(*first++);
        ++p->size_;
    }

    return details::ObjectUnsafe::Downcast<SmallMapImpl>(impl);
}

std::tuple<ObjectPtr<Object>, SmallMapImpl::iterator, bool> SmallMapImpl::InsertImpl(
        KVType&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    auto* map = static_cast<SmallMapImpl*>(old_impl.get());//NOLINT
    if (const auto it = map->find(kv.first); it != map->end()) {
        if (assign) {
            it->second = std::move(kv.second);
        }
        return {old_impl, it, false};
    }

    if (map->size() < map->slots()) {
        auto* p = static_cast<KVType*>(map->data()) + map->size();
        new (p) KVType(std::move(kv));
        ++map->size_;
        return {old_impl, iterator(map->size() - 1, map), true};
    }

    const size_t new_cap = std::min(kThreshold, std::max(kIncFactor * map->slots(), kInitSize));
    auto new_impl = details::ObjectUnsafe::Downcast<SmallMapImpl>(Create(new_cap));
    auto* src = static_cast<KVType*>(map->data());
    auto* dst = static_cast<KVType*>(new_impl->data());
    for (size_t i = 0; i < map->size(); ++i) {
        new (dst++) KVType(std::move(*src++));
        ++new_impl->size_;
    }
    new (dst) KVType(std::move(kv));
    ++new_impl->size_;
    iterator pos(new_impl->size() - 1, new_impl.get());
    return {new_impl, pos, true};
}

SmallMapImpl::iterator SmallMapImpl::erase_impl(iterator pos) {
    if (pos == end()) {
        return end();
    }

    // auto src = end() - 1;
    // *pos = std::move(*src);

    auto src = pos + 1;
    auto dst = pos;
    while (src != end()) {
        *dst = std::move(*src);
        ++src;
        ++dst;
    }
    --size_;
    return pos;
}

struct DenseMapImpl::Entry {
    KVType data{};
    size_t prev = kInvalidIndex;
    size_t next = kInvalidIndex;

    Entry() = default;
    Entry(key_type key, mapped_type value) : data(std::move(key), std::move(value)) {}
    explicit Entry(const KVType& kv) : data(kv) {}
    explicit Entry(KVType&& kv) : data(std::move(kv)) {}

    ~Entry() {
        reset();
    }

    void reset() {
        data.first.reset();
        data.second.reset();
        prev = kInvalidIndex;
        next = kInvalidIndex;
    }
};

struct DenseMapImpl::Block {
    uint8_t bytes[kBlockSize + kBlockSize * sizeof(Entry)];

    Block() {// NOLINT
        auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (uint8_t i = 0; i < kBlockSize; ++i) {
            bytes[i] = kEmptySlot;
            new (data + i) Entry;
        }
    }

    Block(const Block& other) {// NOLINT
        const auto* src = reinterpret_cast<const Entry*>(other.bytes + kBlockSize);
        auto* dst = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (uint8_t i = 0; i < kBlockSize; ++i) {
            bytes[i] = other.bytes[i];
            new (dst + i) Entry(src[i]);
        }
    }

    ~Block() {
        auto* data = reinterpret_cast<Entry*>(bytes + kBlockSize);
        for (uint8_t i = 0; i < kBlockSize; ++i) {
            bytes[i] = kEmptySlot;
            data[i].~Entry();
        }
    }
};

class DenseMapImpl::Cursor {
public:
    Cursor() : index_(0), obj_(nullptr) {}

    Cursor(size_t index, const DenseMapImpl* p) : index_(index), obj_(p) {}

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

    NODISCARD bool IsIterListHead() const {
        return index() == obj()->iter_list_head_;
    }

    NODISCARD bool IsIterListTail() const {
        return index() == obj()->iter_list_tail_;
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

    NODISCARD mapped_type& GetValue() const {
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
    void SetOffsetIdx(uint8_t offset_idx) const {
        CHECK(offset_idx < kNumOffsetDists);
        (GetMeta() &= 0x80) |= offset_idx;
    }

    // Destroy the item in the entry.
    void DestructData() const {
        GetEntry().~Entry();
    }

    // Construct a head of linked list inplace.
    void CreateHead(const Entry& entry) const {
        GetMeta() = 0x00;
        GetEntry() = entry;
    }

    // Construct a tail of linked list inplace
    void CreateTail(const Entry& entry) const {
        GetMeta() = 0x80;
        GetEntry() = entry;
    }

    // Whether the slot has the next slot on the linked list
    NODISCARD bool HasNextSlot() const {
        return NextProbePosOffset[GetMeta() & 0x7F] != 0;
    }

    // Move the current cursor to the next slot on the linked list
    bool MoveToNextSlot(std::optional<uint8_t> meta_opt = std::nullopt) {
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

    // Get the prev slot on the linked list
    NODISCARD Cursor FindPrevSlot() const {
        // start from the head of the linked list, which must exist
        auto cur = obj()->GetCursorFromHash(AnyHash()(GetKey()));
        auto prev = cur;

        cur.MoveToNextSlot();
        while (index() != cur.index()) {
            prev = cur;
            cur.MoveToNextSlot();
        }

        return prev;
    }

    NODISCARD std::optional<std::pair<uint8_t, Cursor>> GetNextEmptySlot() const {
        for (uint8_t i = 1; i < kNumOffsetDists; ++i) {
            if (Cursor candidate((index() + NextProbePosOffset[i]) % obj()->slots(), obj());
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

DenseMapImpl::Block* DenseMapImpl::GetBlock(size_t block_idx) const {
    return static_cast<Block*>(data()) + block_idx;
}

DenseMapImpl::Cursor DenseMapImpl::GetCursorFromHash(size_t hash_value) const {
    return {details::FibonacciHash(hash_value, fib_shift_), this};
}

std::optional<DenseMapImpl::Cursor> DenseMapImpl::GetListHead(size_t hash_value) const {
    if (const auto head = GetCursorFromHash(hash_value); head.IsHead()) {
        return head;
    }

    return std::nullopt;
}

DenseMapImpl::KVType* DenseMapImpl::GetDataPtrImpl(size_t index) const {
    return &Cursor(index, this).GetData();
}

size_t DenseMapImpl::GetNextIndexOfImpl(size_t index) const {
    // keep at the end of iterator
    if (index == kInvalidIndex) {
        return index;
    }

    return Cursor(index, this).GetEntry().next;
}

size_t DenseMapImpl::GetPrevIndexOfImpl(size_t index) const {
    // this is the end iterator, we need to return tail.
    if (index == kInvalidIndex) {
        return iter_list_tail_;
    }

    return Cursor(index, this).GetEntry().prev;
}

size_t DenseMapImpl::count_impl(const key_type& key) const {
    return !Search(key).IsNone();
}

DenseMapImpl::const_iterator DenseMapImpl::find_impl(const key_type& key) const {
    return const_cast<DenseMapImpl*>(this)->find_impl(key);
}

DenseMapImpl::iterator DenseMapImpl::find_impl(const key_type& key) {
    const auto node = Search(key);
    return node.IsNone() ? end() : iterator(node.index(), this);
}

void DenseMapImpl::erase_impl1(const iterator& pos) {
    const auto idx = pos.index();
    if (pos.ptr() == nullptr || idx >= slots()) {
        return;
    }

    if (Cursor cur(idx, this); cur.HasNextSlot()) {
        Cursor prev = cur;
        Cursor last = cur;
        last.MoveToNextSlot();
        while (last.MoveToNextSlot()) {
            prev = last;
        }

        // needs to first unlink node from the list
        IterListRemove(cur);
        // move data from last to node
        cur.GetData() = std::move(last.GetData());
        // Move link chain of iter to last as we store last node to the new iter loc.
        IterListReplace(last, cur);
        // last.DestructData();
        last.SetEmpty();
        prev.SetOffsetIdx(0);
    } else {// the last node
        if (!cur.IsHead()) {
            // cut the link if there is any
            cur.FindPrevSlot().SetOffsetIdx(0);
        }
        // unlink the node from iterator list
        IterListRemove(cur);
        cur.DestructData();
        cur.SetEmpty();
    }
    --size_;
}

DenseMapImpl::iterator DenseMapImpl::erase_impl(iterator pos) {
    if (pos == end()) {
        return end();
    }

    auto nest_pos = pos + 1;

    if (Cursor cur(pos.index(), this); cur.HasNextSlot()) {
        Cursor prev = cur;
        Cursor last = cur;
        last.MoveToNextSlot();
        while (last.MoveToNextSlot()) {
            prev = last;
        }

        // needs to first unlink node from the list
        IterListRemove(cur);
        // move data from last to node
        cur.GetData() = std::move(last.GetData());
        // Move link chain of iter to last as we store last node to the new iter loc.
        IterListReplace(last, cur);
        // last.DestructData();
        last.SetEmpty();
        prev.SetOffsetIdx(0);
    } else {// the last node
        if (!cur.IsHead()) {
            // cut the link if there is any
            cur.FindPrevSlot().SetOffsetIdx(0);
        }
        // unlink the node from iterator list
        IterListRemove(cur);
        cur.DestructData();
        cur.SetEmpty();
    }
    --size_;
    return nest_pos;
}

DenseMapImpl::mapped_type& DenseMapImpl::At(const key_type& key) const {
    const Cursor iter = Search(key);
    if (iter.IsNone()) {
        AETHERMIND_THROW(KeyError) << "Key not found";
    }

    return iter.GetValue();
}

DenseMapImpl::Cursor DenseMapImpl::Search(const key_type& key) const {
    if (empty()) {
        return {};
    }

    const auto head_opt = GetListHead(AnyHash()(key));
    if (!head_opt.has_value()) {
        return {};
    }

    auto node = *head_opt;
    while (!node.IsNone()) {
        if (key == node.GetKey()) {
            return node;
        }
        node.MoveToNextSlot();
    }

    return {};
}

void DenseMapImpl::reset() {
    const size_t block_num = ComputeBlockNum(slots());
    auto* p = static_cast<Block*>(data());
    for (size_t i = 0; i < block_num; ++i) {
        p[i].~Block();
    }

    size_ = 0;
    slots_ = 0;
    fib_shift_ = 63;
}

void DenseMapImpl::IterListPushBack(Cursor node) {
    node.GetEntry().prev = iter_list_tail_;
    node.GetEntry().next = kInvalidIndex;

    if (iter_list_head_ == kInvalidIndex && iter_list_tail_ == kInvalidIndex) {
        iter_list_head_ = node.index();
    } else {
        Cursor(iter_list_tail_, this).GetEntry().next = node.index();
    }

    iter_list_tail_ = node.index();
}

void DenseMapImpl::IterListRemove(Cursor node) {
    // head
    if (node.IsIterListHead()) {
        iter_list_head_ = node.GetEntry().next;
    } else {
        Cursor prev_node(node.GetEntry().prev, this);
        prev_node.GetEntry().next = node.GetEntry().next;
    }

    // tail
    if (node.IsIterListTail()) {
        iter_list_tail_ = node.GetEntry().prev;
    } else {
        Cursor next_node(node.GetEntry().next, this);
        next_node.GetEntry().prev = node.GetEntry().prev;
    }
}

void DenseMapImpl::IterListReplace(Cursor src, Cursor dst) {
    dst.GetEntry().prev = src.GetEntry().prev;
    dst.GetEntry().next = src.GetEntry().next;

    if (src.IsIterListHead()) {
        iter_list_head_ = dst.index();
    } else {
        Cursor prev_node(dst.GetEntry().prev, this);
        prev_node.GetEntry().next = dst.index();
    }

    if (src.IsIterListTail()) {
        iter_list_tail_ = dst.index();
    } else {
        Cursor next_node(dst.GetEntry().next, this);
        next_node.GetEntry().prev = dst.index();
    }
}

std::optional<DenseMapImpl::Cursor> DenseMapImpl::TrySpareListHead(Cursor target, KVType&& kv) {
    // `target` is not the head of the linked list
    // move the original item of `target` (if any)
    // and construct new item on the position `target`
    // To make `target` empty, we
    // 1) find `w` the previous element of `target` in the linked list
    // 2) copy the linked list starting from `r = target`
    // 3) paste them after `w`

    // read from the linked list after `r`
    Cursor r = target;
    // write to the tail of `w`
    Cursor w = target.FindPrevSlot();
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    uint8_t r_meta;

    do {
        const auto empty_slot_info = w.GetNextEmptySlot();
        if (!empty_slot_info) {
            return std::nullopt;
        }

        uint8_t offset_idx = empty_slot_info->first;
        Cursor empty = empty_slot_info->second;

        // move `r` to `empty`
        // first move the data over
        empty.CreateTail(Entry(std::move(r.GetData())));
        // then move link list chain of r to empty
        // this needs to happen after NewTail so empty's prev/next get updated
        IterListReplace(r, empty);
        // clear the metadata of `r`
        r_meta = r.GetMeta();
        if (is_first) {
            is_first = false;
            r.SetProtected();
        } else {
            r.SetEmpty();
        }
        // link `w` to `empty`, and move forward
        w.SetOffsetIdx(offset_idx);
        w = empty;
    } while (r.MoveToNextSlot(r_meta));// move `r` forward as well

    // finally, we have done moving the linked list
    // fill data_ into `target`
    target.CreateHead(Entry(std::move(kv)));
    ++size_;
    return target;
}

std::pair<DenseMapImpl::iterator, bool> DenseMapImpl::TryInsert(KVType&& kv, bool assign) {
    // The key is already in the hash table
    if (auto it = find(kv.first); it != end()) {
        if (assign) {
            Cursor cur(it.index(), it.ptr());
            cur.GetValue() = std::move(kv.second);
            IterListRemove(cur);
            IterListPushBack(cur);
        }
        return {it, false};
    }

    // required that `iter` to be the head of a linked list through which we can iterator.
    // `iter` can be:
    // 1) empty;
    // 2) body of an irrelevant list;
    // 3) head of the relevant list.
    auto node = GetCursorFromHash(AnyHash()(kv.first));

    // Case 1: empty
    if (node.IsEmpty()) {
        node.CreateHead(Entry(std::move(kv)));
        ++size_;
        IterListPushBack(node);
        return {iterator(node.index(), this), true};
    }

    // Case 2: body of an irrelevant list
    if (!node.IsHead()) {
        if (IsFull()) {
            return {end(), false};
        }

        if (const auto target = TrySpareListHead(node, std::move(kv)); target.has_value()) {
            IterListPushBack(target.value());
            return {iterator(target->index(), this), true};
        }
        return {end(), false};
    }

    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    // make sure `iter` is the prev element of `cur`
    Cursor cur = node;
    while (cur.MoveToNextSlot()) {
        // make sure `node` is the previous element of `cur`
        node = cur;
    }

    // `node` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
        return {end(), false};
    }

    // find the next empty slot
    auto empty_slot_info = node.GetNextEmptySlot();
    if (!empty_slot_info) {
        return {end(), false};
    }

    uint8_t offset_idx = empty_slot_info->first;
    Cursor empty = empty_slot_info->second;
    empty.CreateTail(Entry(std::move(kv)));
    // link `iter` to `empty`, and move forward
    node.SetOffsetIdx(offset_idx);
    IterListPushBack(empty);
    ++size_;
    return {iterator(node.index(), this), true};
}

std::tuple<ObjectPtr<Object>, DenseMapImpl::iterator, bool> DenseMapImpl::InsertImpl(
        KVType&& kv, const ObjectPtr<Object>& old_impl, bool assign) {
    auto* map = static_cast<DenseMapImpl*>(old_impl.get());// NOLINT
    if (auto [it, is_success] = map->TryInsert(std::move(kv), assign); it != map->end()) {
        return {old_impl, it, is_success};
    }

    // Otherwise, start rehash
    auto new_impl = details::ObjectUnsafe::Downcast<DenseMapImpl>(
            Create(map->slots() * kIncFactor));

    // need to insert in the same order as the original map
    size_t idx = map->iter_list_head_;
    while (idx != kInvalidIndex) {
        Cursor cur(idx, map);
        new_impl->TryInsert(std::move(cur.GetData()), assign);
        idx = cur.GetEntry().next;
    }

    auto [pos, is_success] = new_impl->TryInsert(std::move(kv), assign);
    return {new_impl, pos, is_success};
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

ObjectPtr<DenseMapImpl> DenseMapImpl::CreateImpl(size_t n) {
    CHECK(n > kThreshold);
    auto [fib_shift, slots] = ComputeSlotNum(n);
    const size_t block_num = ComputeBlockNum(slots);
    auto impl = make_array_object<DenseMapImpl, Block>(block_num);
    impl->data_ = reinterpret_cast<char*>(impl.get()) + sizeof(DenseMapImpl);
    impl->size_ = 0;
    impl->slots_ = slots;
    impl->fib_shift_ = fib_shift;
    impl->iter_list_head_ = kInvalidIndex;
    impl->iter_list_tail_ = kInvalidIndex;

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        new (p + i) Block;
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

    auto* p = static_cast<Block*>(impl->data());
    for (size_t i = 0; i < block_num; ++i) {
        new (p + i) Block(*src->GetBlock(i));
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
