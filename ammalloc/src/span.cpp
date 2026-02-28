//
// Created by richard on 2/7/26.
//

#include "ammalloc/span.h"

namespace aethermind {

void Span::Init(size_t object_size) {
    obj_size = object_size;

    // 1. Calculate Base Address
    void* start_ptr = details::PageIDToPtr(start_page_idx);
    const size_t total_bytes = page_num << SystemConfig::PAGE_SHIFT;

    // 2. Estimate Bitmap Size
    // Formula: Total = BitmapBytes + DataBytes
    //          Total = (Num * 1/8) + (Num * ObjSize)
    size_t max_objs = (total_bytes * 8) / (obj_size * 8 + 1);
    bitmap_num = (max_objs + 64 - 1) >> 6;
    // Placement New: Create atomic array at page start
    bitmap = new (start_ptr) uint64_t[bitmap_num];

    // 3. Calculate Data Start Address (Aligned)
    uintptr_t data_start = reinterpret_cast<uintptr_t>(bitmap) + bitmap_num * 8;
    data_start = (data_start + 16 - 1) & ~(16 - 1);
    data_base_ptr = reinterpret_cast<void*>(data_start);

    // 4. Calculate Actual Capacity
    uintptr_t data_end = reinterpret_cast<uintptr_t>(start_ptr) + total_bytes;
    if (data_start >= data_end) {
        capacity = 0;
    } else {
        capacity = (data_end - data_start) / obj_size;
    }

    // 5. Initialize Bitmap Bits (Loop unrolling for performance)
    size_t full_bitmap_num = capacity / 64;
    size_t tail_bits = capacity & 63;

    // Part A: Set full blocks to ~0ULL(all free)
    for (size_t i = 0; i < full_bitmap_num; ++i) {
        bitmap[i] = ~0ULL;
    }

    // Part B: Handle the tail block(if any)
    if (full_bitmap_num < bitmap_num) {
        if (tail_bits == 0) {
            // If capacity is exact multiple of 64, this block is actually out of bounds
            // But if num_full_blocks == bitmap_num, we won't enter this branch unless i < bitmap_num
            // So this handles the case where capacity < bitmap_num * 64
            bitmap[full_bitmap_num] = 0;
        } else {
            // Set lower 'tail_bits' to 1, rest to 0
            uint64_t mask = (1ULL << tail_bits) - 1;
            // Relaxed is sufficient during initialization
            bitmap[full_bitmap_num] = mask;
        }

        // Part C: Zero out remaining padding blocks
        for (size_t i = full_bitmap_num + 1; i < bitmap_num; ++i) {
            // Out of capacity range blocks (padding space)
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

    size_t idx = scan_cursor;
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
        return static_cast<char*>(data_base_ptr) + global_obj_idx * obj_size;
    }
    return nullptr;
}

void Span::FreeObject(void* ptr) {
    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(data_base_ptr);
    size_t global_obj_idx = offset / obj_size;

    size_t bitmap_idx = global_obj_idx >> 6;
    int bit_pos = global_obj_idx & (64 - 1);
    bitmap[bitmap_idx] |= (1ULL << bit_pos);
    --use_count;
}

}// namespace aethermind
