# ammalloc Comprehensive Test & Benchmark Plan

## 1. Executive Summary

### 1.1 Objective
Build a comprehensive testing and benchmarking suite for ammalloc that:
- Validates functional correctness across single-threaded and multi-threaded scenarios
- Provides fair performance comparisons against std::malloc, TCMalloc, and jemalloc
- Measures memory efficiency (RSS, fragmentation, page release behavior)
- Establishes reproducible methodology for continuous regression detection

### 1.2 Success Criteria
- [ ] All correctness tests pass under TSan (ThreadSanitizer)
- [ ] Benchmark results show statistical significance (CV < 5% for latency tests)
- [ ] Cross-allocator comparison produces actionable insights
- [ ] CI integration prevents performance regressions > 10%

### 1.3 Constraints & Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| HugePageCache uses std::vector (P0 bug) | Crash in interposition mode | Fix before full interposition testing |
| Bootstrap recursion | Deadlock | Implement in_malloc guard or stick to API mode initially |
| Cross-allocator contamination | Biased results | Use separate processes, verify with dladdr |

---

## 2. Technical Architecture

### 2.1 Comparison Modes

#### Mode A: API Mode (Primary, Immediate)
Direct function pointer dispatch to allocator-specific APIs.

```cpp
template<typename AllocFn, typename FreeFn>
class AllocatorHarness {
    AllocFn alloc_;
    FreeFn free_;
public:
    void* allocate(size_t n) { return alloc_(n); }
    void deallocate(void* p) { free_(p); }
};

// Instantiations
using AmmallocHarness = AllocatorHarness<am_malloc, am_free>;
using StdHarness = AllocatorHarness<std::malloc, std::free>;
using JemallocHarness = AllocatorHarness<je_malloc, je_free>;  // or malloc if interposing
using TcmallocHarness = AllocatorHarness<tc_malloc, tc_free>;  // or malloc if interposing
```

**Advantages:**
- Safe, no bootstrap issues
- Precise control over code paths
- Can run all allocators in single binary (sequential tests)

**Disadvantages:**
- Misses C++ new/delete paths
- "Other heap" allocations still use system allocator

#### Mode B: Full Interposition (Secondary, Future)
Build separate executables with allocator linked at build time.

```cmake
add_executable(bench_ammalloc ...)  # Links ammalloc
add_executable(bench_system ...)     # Links libc malloc
add_executable(bench_jemalloc ...)   # Links -ljemalloc
add_executable(bench_tcmalloc ...)   # Links -ltcmalloc
```

**Prerequisites:**
1. Remove std::vector from HugePageCache
2. Implement recursion-safe initialization
3. Provide complete malloc/free/calloc/realloc/posix_memalign/aligned_alloc
4. Override C++ new/delete operators

### 2.2 Test Harness Design Principles

**Anti-Contamination Measures:**
- Pre-allocate all test structures (vectors, queues) before timing
- Use fixed-size arrays on stack where possible
- Avoid std::string, iostreams, logging during timed sections
- Use `benchmark::DoNotOptimize()` and `benchmark::ClobberMemory()`

**Statistical Rigour:**
- Fixed RNG seeds (reproducible)
- Warmup phase (populate thread caches)
- Minimum 5 repetitions
- Report: mean, stddev, CV%, 95% CI

---

## 3. Test Taxonomy

### 3.1 Correctness Tests (Non-Timed)

#### 3.1.1 Single-Threaded
| Test Case | Description | Validation |
|-----------|-------------|------------|
| BasicAllocFree | 8B-32KB all size classes | Read/write pattern verification |
| ZeroSize | malloc(0) behavior | Consistency across allocators |
| NullFree | free(nullptr) | Must not crash |
| DoubleFree | Detect corruption | Bitmap state check |
| LargeAlloc | >32KB, >512KB, >1MB | Direct mmap path |
| Alignment | alignof(max_align_t) compliance | Pointer alignment verification |

#### 3.1.2 Multi-Threaded
| Test Case | Thread Pattern | Validation |
|-----------|---------------|------------|
| ConcurrentAllocFree | N threads, independent | No data races (TSan) |
| CrossThreadFree | Alloc on T0, free on T1 | TransferCache correctness |
| ProducerConsumer | Ring buffer pattern | No ABA issues |
| BurstAllocation | Synchronized start | Scalability measurement |

### 3.2 Performance Microbenchmarks

#### 3.2.1 Latency Tests (Single-Threaded)

| Benchmark | Size | Window | What It Measures |
|-----------|------|--------|------------------|
| Churn_FastPath | 8B, 64B, 256B, 4KB | 1 | Pure alloc+free latency |
| Churn_TC_Steady | 8B, 64B | 256 | ThreadCache hot path |
| Churn_CC_Pressure | 8B, 64B | 1024 | CentralCache interaction |
| Churn_Deep | 8B, 64B | 2000+ | Span allocation pressure |
| RandomSize | 1B-32KB uniform | 1024 | Size class distribution |
| LargeObject | 64KB, 512KB, 1MB | 8 | Direct mmap overhead |

#### 3.2.2 Throughput Tests (Multi-Threaded)

| Benchmark | Size | Batch | Threads | Metric |
|-----------|------|-------|---------|--------|
| MT_Parallel | 8B, 64B, 1KB | 1000 | 1,2,4,8,16,32,64 | Ops/sec, Wall time |
| MT_Contended | 64B | 1000 | 1,2,4,8,16 | Lock contention effects |
| MT_Random | Mixed | 1000 | 16 | Realistic workload |

#### 3.2.3 Lifetime Patterns

| Pattern | Description | Allocator Stress |
|---------|-------------|------------------|
| LIFO | Stack order | ThreadCache optimal |
| FIFO | Queue order | CentralCache churn |
| Random | Random free order | Fragmentation stress |
| Generational | Young/old objects | Page release behavior |

### 3.3 Memory Efficiency Tests

#### 3.3.1 Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| RSS | Resident Set Size | `/proc/self/statm` (Linux) |
| Peak RSS | Maximum resident memory | `getrusage(RUSAGE_SELF)` |
| Internal Fragmentation | `usable - requested` | `malloc_usable_size()` |
| External Fragmentation | `allocated - used` | Live bytes / Committed bytes |
| Page Release Rate | RSS reduction over time | Time-series sampling |

#### 3.3.2 Test Scenarios

```cpp
void BM_MemoryLifecycle(benchmark::State& state) {
    // Phase 1: Baseline RSS
    // Phase 2: Ramp to peak (allocate N objects)
    // Phase 3: Fragmentation (free 50% randomly)
    // Phase 4: Reallocation (attempt large allocation)
    // Phase 5: Full release (free all)
    // Phase 6: Steady state (measure page release)
    
    state.counters["rss_baseline_mb"] = baseline;
    state.counters["rss_peak_mb"] = peak;
    state.counters["rss_final_mb"] = final;
    state.counters["fragmentation_pct"] = frag;
}
```

### 3.4 PageHeapScavenger Tests

| Test | Description |
|------|-------------|
| ScavengerStartup | Verify thread starts on first slow path |
| IdlePageRelease | Measure MADV_DONTNEED effectiveness |
| ReleaseLatency | Time from free to RSS reduction |
| PressureResponse | High load vs idle behavior |

---

## 4. Implementation Plan

### 4.1 Phase 1: Foundation (Week 1)

**Tasks:**
1. [ ] Fix HugePageCache std::vector → fixed array
2. [ ] Implement BM_MemoryFootprint with RSS tracking
3. [ ] Add jemalloc/TCMalloc API wrappers
4. [ ] Create correctness test suite (single-threaded)

**Deliverables:**
- `tests/benchmark/benchmark_memory_efficiency.cpp`
- `tests/unit/test_allocator_correctness.cpp`
- CMake integration for jemalloc/tcmalloc detection

### 4.2 Phase 2: Completeness (Week 2)

**Tasks:**
1. [ ] Implement cross-thread free tests
2. [ ] Add lifetime pattern benchmarks
3. [ ] Create PageHeapScavenger efficiency tests
4. [ ] Build automated comparison reports

**Deliverables:**
- `tests/benchmark/benchmark_lifecycle_patterns.cpp`
- `tests/benchmark/benchmark_page_release.cpp`
- `tools/generate_comparison_report.py`

### 4.3 Phase 3: CI Integration (Week 3)

**Tasks:**
1. [ ] GitHub Actions workflow for benchmarks
2. [ ] Performance regression detection
3. [ ] Automated report generation on PR

**Deliverables:**
- `.github/workflows/benchmark.yml`
- Performance dashboard (GitHub Pages or artifact)

### 4.4 Phase 4: Full Interposition (Future)

**Prerequisites:** Bootstrap-safe ammalloc

**Tasks:**
1. [ ] Implement complete malloc family
2. [ ] Add new/delete overrides
3. [ ] Build separate executables per allocator
4. [ ] Implement dladdr verification

---

## 5. Configuration Standards

### 5.1 jemalloc Configurations

```bash
# High throughput
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# Low memory
export MALLOC_CONF="background_thread:true,tcache_max:4096,dirty_decay_ms:5000,muzzy_decay_ms:5000"
```

### 5.2 TCMalloc Configurations

```bash
export TCMALLOC_RELEASE_RATE=10.0
export TCMALLOC_MAX_PER_CPU_CACHE_SIZE=33554432  # 32MB
```

### 5.3 ammalloc Configurations

```bash
export AM_ENABLE_SCAVENGER=1
export AM_SCAVENGER_INTERVAL_MS=1000
export AM_TC_SIZE=32768
```

---

## 6. Data Collection & Reporting

### 6.1 Output Formats

**JSON (Raw Data):**
```json
{
  "benchmark": "BM_Churn_8_1",
  "allocator": "ammalloc",
  "mean_ns": 3.52,
  "stddev_ns": 0.07,
  "cv_pct": 1.96,
  "rss_peak_mb": 45.2,
  "throughput_ops_per_sec": 284090909
}
```

**TSV (Analysis):**
```tsv
benchmark	allocator	mean_ns	stddev_ns	cv_pct	rss_peak_mb	speedup_vs_std
BM_Churn_8_1	ammalloc	3.52	0.07	1.96	45.2	1.91
BM_Churn_8_1	std	6.71	0.07	1.04	52.1	1.00
```

**Markdown (Report):**
```markdown
## Summary

ammalloc shows 1.91x speedup over std::malloc for small object allocation.

| Metric | ammalloc | std | jemalloc | tcmalloc |
|--------|----------|-----|----------|----------|
| Latency (ns) | 3.52 | 6.71 | 4.12 | 3.89 |
| RSS Peak (MB) | 45.2 | 52.1 | 48.7 | 47.3 |
```

### 6.2 Visualization

- Latency distribution (box plots)
- Throughput scaling curves (threads vs ops/sec)
- Memory usage over time (lifecycle graphs)
- Speedup heatmaps (size vs threads)

---

## 7. Open Questions (To Be Resolved)

1. **Platform Scope**: Linux-only for official numbers? macOS as secondary?
2. **jemalloc Integration**: Prefixed APIs (`je_malloc`) vs interposing builds?
3. **CI Depth**: PMU counters in CI or manual only?
4. **Acceptable Regression Threshold**: 5%, 10%, or size-dependent?

---

## 8. Acceptance Criteria

### 8.1 Functional
- [ ] All correctness tests pass with TSan enabled
- [ ] No memory leaks detected by ASan
- [ ] Cross-thread free produces no data races

### 8.2 Performance
- [ ] Benchmark CV < 5% for latency tests (after warmup)
- [ ] Benchmark CV < 10% for multithreaded tests
- [ ] Statistical significance (p < 0.05) for performance differences

### 8.3 Comparability
- [ ] Identical workloads across all allocators
- [ ] Documented configuration for each allocator
- [ ] Baseline subtraction for memory metrics

### 8.4 Automation
- [ ] CI runs benchmarks on every PR
- [ ] Regression alerts for > 10% slowdown
- [ ] Automated report generation and archival

---

## 9. Appendix

### 9.1 File Structure

```
tests/
├── benchmark/
│   ├── benchmark_memory_pool.cpp          # Existing
│   ├── benchmark_memory_efficiency.cpp    # NEW
│   ├── benchmark_lifecycle_patterns.cpp   # NEW
│   ├── benchmark_page_release.cpp         # NEW
│   └── CMakeLists.txt
├── unit/
│   ├── test_allocator_correctness.cpp     # NEW
│   └── ...
├── correctness/                           # NEW dir
│   ├── test_single_thread.cpp
│   ├── test_multi_thread.cpp
│   └── test_cross_thread.cpp
tools/
├── plot_ammalloc_benchmark.py             # Existing
├── generate_comparison_report.py          # NEW
└── run_benchmark_suite.sh                 # NEW
```

### 9.2 References

1. `docs/ammalloc_benchmark_rigorous_20260303.md` - Existing methodology
2. `docs/allocator_background_thread_research.md` - jemalloc/tcmalloc research
3. `docs/review.md` - Known issues (HugePageCache P0)
4. `docs/ammalloc_todo_list.md` - Bootstrap recursion fix
5. jemalloc TUNING.md: https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md
6. TCMalloc Tuning: https://google.github.io/tcmalloc/tuning.html
7. Google Benchmark Best Practices: https://github.com/google/benchmark

---

*Plan Version: 1.0*
*Created: 2026-03-06*
*Status: Decision-Complete (Pending Self-Review)*
