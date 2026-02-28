# AetherMind Memory Allocator (ammalloc) - AI Context Document

## 🤖 AI Assistant Instructions
When assisting with the `ammalloc` module, you must act as a **Senior C++ Systems Architect**. 
This module is an ultra-high-performance, concurrent memory allocator designed to replace the system `malloc/free`. 
**Performance (nanosecond-level latency), concurrency (linear scalability), and memory safety are the highest priorities.**

Before generating any code or suggestions, you MUST adhere to the constraints and architectural rules defined in this document.

---

## 🏗️ Architecture Overview
`ammalloc` follows a 3-tier caching architecture inspired by Google TCMalloc:

1. **ThreadCache (Frontend)**
   - **Type**: Thread Local Storage (TLS).
   - **Locking**: **Completely Lock-Free**.
   - **Role**: Handles the vast majority of allocations/deallocations. Uses embedded `FreeList` (LIFO) and a Slow-Start dynamic quota mechanism.
2. **CentralCache (Middle-end)**
   - **Type**: Global Singleton with fine-grained Bucket Locks.
   - **Locking**: `SpinLock` (TTAS) for Fast Path, `std::mutex` for Slow Path.
   - **Role**: Balances memory between threads. Uses a `TransferCache` (array of pointers) for ultra-fast batch transfers, and falls back to `SpanList` (Bitmap scanning) when exhausted. Incorporates a **Prefetching** mechanism.
3. **PageCache (Backend)**
   - **Type**: Global Singleton.
   - **Locking**: Global `std::mutex`.
   - **Role**: Manages physical pages. Handles Span splitting and coalescing (merging adjacent free spans). Interacts with the OS via `PageAllocator`.

---

## 🚫 Strict Engineering Constraints (NEVER VIOLATE)

### 1. The Bootstrapping Problem (No `malloc` recursion)
- **Rule**: NEVER use standard STL containers (`std::vector`, `std::string`, `std::map`, etc.) or the `new`/`delete` operators inside the core allocation/deallocation paths or metadata management.
- **Reason**: Using them will trigger the system `malloc`, which is hooked by `am_malloc`, leading to infinite recursion and Stack Overflow.
- **Solution**: Use fixed-size stack arrays, embedded linked lists, or the custom `ObjectPool` for metadata (e.g., `Span`, `RadixNode`).

### 2. Cache Locality & False Sharing
- **Rule**: Core structures accessed concurrently by different threads (e.g., `ThreadCache`, `CentralCache::Bucket`) MUST be aligned to the cache line size using `alignas(SystemConfig::CACHE_LINE_SIZE)`.
- **Rule**: Always preserve **LIFO (Last-In-First-Out)** order when moving pointers between caches to maximize CPU L1/L2 cache hit rates.

### 3. Concurrency & Memory Ordering
- **Rule**: Fast paths must remain lock-free.
- **Rule**: When using `std::atomic`, always specify the exact memory order. 
  - Use `std::memory_order_relaxed` for counters or hints where exact synchronization isn't critical.
  - Use `std::memory_order_acquire` and `std::memory_order_release` strictly for publishing/consuming shared memory (e.g., RadixTree nodes, Bitmap bits).

### 4. Radix Tree (PageMap) Integrity
- **Rule**: The `PageMap` is a 4-level Radix Tree covering a 48-bit virtual address space.
- **Rule**: Readers (`GetSpan`) are lock-free. Writers (`SetSpan`) are protected by the `PageCache` mutex.
- **Rule**: NEVER explicitly delete a `RadixNode`. The tree only grows. Memory is reclaimed only upon full system shutdown via `ObjectPool::ReleaseMemory`.

---

## 🧠 Key Design Decisions & "Why"s

*   **Why `TransferCache`?** 
    Scanning Bitmaps in `CentralCache` under a mutex is too slow for high-concurrency batch operations. `TransferCache` reduces this to $O(1)$ array copies under a lightweight `SpinLock`.
*   **Why Prefetching in `CentralCache`?**
    When a thread hits the slow path (Span Bitmap scanning), it extracts extra objects to refill the `TransferCache` for the *next* thread, amortizing the mutex cost.
*   **Why 4-Level Radix Tree?**
    To support 64-bit OS ASLR (Address Space Layout Randomization) without allocating massive flat arrays. It handles sparse, high-memory addresses efficiently.
*   **Why Optimistic Huge Pages?**
    `PageAllocator` attempts to allocate exact sizes first. If the OS returns a 2MB-aligned address by chance (THP), we keep it. This avoids the VMA fragmentation and syscall overhead of the "Over-allocate & Trim" fallback.
*   **Why `MADV_DONTNEED` instead of `munmap` for HugePageCache?**
    It releases physical memory (reducing RSS) but keeps the Virtual Memory Area (VMA) intact, allowing subsequent allocations to bypass the heavy `mmap_sem` kernel lock.

---

## 📊 Performance Baselines (For Reference)
When suggesting optimizations, ensure they do not degrade these baselines (Tested on 16-core CPU):
- **Single-Thread Fast Path**: ~3.8 ns
- **Random Size Allocation**: ~26.0 ns (O(1) compile-time lookup)
- **16-Thread High Contention (64B)**: ~8.9 µs / 100+ GiB/s throughput.

## 🛠️ Tech Stack & Conventions
- **Standard**: C++20 (Uses `consteval`, `<bit>`, `std::bit_width`, `std::countr_zero`).
- **Branch Prediction**: Heavily utilizes `AM_LIKELY` and `AM_UNLIKELY` macros.
- **Static vs Runtime Config**: Hard limits (array sizes) are `constexpr` in `StaticConfig`. Soft limits (thresholds) are in `RuntimeConfig` (Singleton).