#include "ammalloc/span.h"

#include <limits>

namespace aethermind {

void Span::Init(size_t object_size) {
    AM_DCHECK(object_size > 0 && object_size <= std::numeric_limits<uint32_t>::max());
    obj_size = static_cast<uint32_t>(object_size);
    void* start_ptr = details::PageIDToPtr(start_page_idx);
    const size_t total_bytes = page_num << SystemConfig::PAGE_SHIFT;

    // Estimate bitmap + data layout:
    // Total = BitmapBytes(1 bit per object) + DataBytes(obj_size per object)
    size_t max_objs = (total_bytes * 8) / (obj_size * 8 + 1);
    size_t bitmap_num = (max_objs + 64 - 1) >> 6;
    auto* bitmap = new (start_ptr) uint64_t[bitmap_num];

    // Calculate data start with alignment padding.
    uintptr_t data_start = reinterpret_cast<uintptr_t>(bitmap) + bitmap_num * 8;
    data_start = (data_start + SystemConfig::ALIGNMENT - 1) & ~(SystemConfig::ALIGNMENT - 1);
    obj_offset = static_cast<uint32_t>(data_start - reinterpret_cast<uintptr_t>(start_ptr));

    // Capacity may be less than max_objs due to alignment overhead.
    uintptr_t data_end = reinterpret_cast<uintptr_t>(start_ptr) + total_bytes;
    capacity = (data_start >= data_end) ? 0 : (data_end - data_start) / obj_size;

    // Initialize bitmap: set first 'capacity' bits to 1 (free).
    size_t full_bitmap_num = capacity / 64;
    size_t tail_bits = capacity & 63;
    for (size_t i = 0; i < full_bitmap_num; ++i) {
        bitmap[i] = ~0ULL;
    }

    if (full_bitmap_num < bitmap_num) {
        bitmap[full_bitmap_num] = (tail_bits == 0) ? 0 : ((1ULL << tail_bits) - 1);
        for (size_t i = full_bitmap_num + 1; i < bitmap_num; ++i) {
            bitmap[i] = 0;
        }
    }

    use_count = 0;
    scan_cursor = 0;
}

void* Span::AllocObject() {
    if (use_count >= capacity) {
        return nullptr;
    }

    auto idx = scan_cursor;
    auto bitmap_num = GetBitmapNum();
    auto* bitmap = GetBitmap();
    for (size_t i = 0; i < bitmap_num; ++i) {
        size_t cur_idx = idx + i;
        if (cur_idx >= bitmap_num) {
            cur_idx -= bitmap_num;
        }

        uint64_t val = bitmap[cur_idx];
        if (val == 0) {
            continue;
        }

        int bit_pos = std::countr_zero(val);
        bitmap[cur_idx] &= ~(1ULL << bit_pos);
        ++use_count;
        scan_cursor = cur_idx;
        size_t global_obj_idx = cur_idx * 64 + bit_pos;
        return static_cast<char*>(GetDataBasePtr()) + global_obj_idx * obj_size;
    }
    return nullptr;
}

void Span::FreeObject(void* ptr) {
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(GetDataBasePtr());
    size_t global_obj_idx = offset / obj_size;

    size_t bitmap_idx = global_obj_idx >> 6;
    int bit_pos = global_obj_idx & (64 - 1);
    uint64_t mask = 1ULL << bit_pos;
    auto* bitmap = GetBitmap();
    AM_DCHECK((bitmap[bitmap_idx] & mask) == 0, "double free detected.");
    bitmap[bitmap_idx] |= mask;
    --use_count;
}

}// namespace aethermind
