# amstring Exception Safety

## 1. Exception Safety Guarantees

### 1.1 Guarantee Levels

| Level | Description |
|-------|-------------|
| **No-throw** | Operation never throws (noexcept) |
| **Strong** | On failure, object unchanged |
| **Basic** | On failure, object in valid state |
| **None** | No guarantee |

### 1.2 Guarantee by Operation

| Operation | Guarantee | Notes |
|-----------|-----------|-------|
| Default constructor | No-throw | Always succeeds |
| Copy constructor | Strong | Allocation failure throws |
| Move constructor | No-throw | noexcept |
| Destructor | No-throw | noexcept |
| Copy assignment | Strong | Allocate before destroy |
| Move assignment | No-throw | noexcept (allocator equal) |
| `data/size/c_str` | No-throw | noexcept |
| `reserve` | Strong | Failure = unchanged |
| `resize` | Strong | Failure = unchanged |
| `clear` | No-throw | noexcept |
| `push_back` | Strong | Failure = unchanged |
| `append` | Strong | Failure = unchanged |
| `insert` | Strong | Failure = unchanged |
| `erase` | No-throw | noexcept |
| `replace` | Strong | Failure = unchanged |

---

## 2. Strong Exception Safety Pattern

### 2.1 Allocate-Copy-Commit

```cpp
void append(const CharT* src, size_type n) {
    size_type new_size = size() + n;
    size_type new_cap = growth_policy::grow(capacity(), new_size);
    
    // Step 1: Allocate new buffer (may throw)
    CharT* new_data = allocate(new_cap + 1);
    
    // Step 2: Copy existing content (may throw)
    char_copy(new_data, data(), size());
    
    // Step 3: Append new content (may throw)
    char_copy(new_data + size(), src, n);
    new_data[new_size] = CharT{};  // null terminator
    
    // Step 4: Commit (no-throw)
    destroy_heap();  // Free old buffer
    set_heap(new_data, new_size, new_cap);  // Update metadata
}
```

Key: Steps 1-3 can throw, Step 4 cannot. If any step throws, original object untouched.

### 2.2 Self-Reference Handling

```cpp
void append(const CharT* src, size_type n) {
    // Check if src points into our buffer
    if (src >= data() && src < data() + size()) {
        // Copy to temporary first (strong guarantee)
        std::vector<CharT> temp(src, src + n);
        append(temp.data(), temp.size());
        return;
    }
    // Normal append
    ...
}
```

---

## 3. noexcept Operations

All these must be noexcept:

```cpp
basic_string() noexcept;
basic_string(basic_string&&) noexcept;
~basic_string() noexcept;

basic_string& operator=(basic_string&&) noexcept;
void clear() noexcept;
void swap(basic_string&) noexcept;

size_type size() const noexcept;
size_type capacity() const noexcept;
bool empty() const noexcept;
const CharT* data() const noexcept;
const CharT* c_str() const noexcept;

CharT& operator[](size_type) noexcept;
CharT& front() noexcept;
CharT& back() noexcept;

iterator erase(const_iterator) noexcept;  // shrinking only
void pop_back() noexcept;  // shrinking only
```

---

## 4. Failing Allocator Tests

### 4.1 Failing Allocator Design

```cpp
template<typename T>
class FailingAllocator {
    size_t fail_after_;  // Fail after N allocations
    size_t alloc_count_;
    
    T* allocate(size_t n) {
        if (++alloc_count_ > fail_after_) {
            throw std::bad_alloc();
        }
        return std::allocator<T>().allocate(n);
    }
};
```

### 4.2 Test Cases

| Test | Purpose |
|------|---------|
| Constructor fail | Empty string on failure |
| Copy fail | Original unchanged |
| Reserve fail | Capacity unchanged |
| Append fail | String unchanged |
| Insert fail | String unchanged |
| Replace fail | String unchanged |

### 4.3 Test Pattern

```cpp
TEST(ExceptionSafety, AppendFail) {
    FailingAllocator<char> alloc(1);  // Fail after 1 allocation
    aethermind::basic_string<char, std::char_traits<char>, FailingAllocator<char>> s(alloc);
    s = "hello";  // First allocation succeeds
    
    EXPECT_THROW(s.append("world"), std::bad_alloc);
    
    // Verify: s unchanged
    EXPECT_EQ(s, "hello");
    EXPECT_EQ(s.size(), 5);
}
```

---

## 5. Allocation Failure Recovery

### 5.1 Recovery Strategy

On allocation failure:
1. Catch `std::bad_alloc`
2. Do not modify original object
3. Let exception propagate

### 5.2 State Validation

After failure, object must be valid:
- `data() != nullptr`
- `data()[size()] == CharT{}`
- `size() <= capacity()`

---

## 6. noexcept(false) Cases

### 6.1 Move Assignment with Unequal Allocator

```cpp
basic_string& operator=(basic_string&& other) noexcept(
    std::allocator_traits<Allocator>::is_always_equal::value ||
    std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value
)
```

If allocators differ and don't propagate, move requires allocation (may throw).

### 6.2 Swap with Unequal Allocator

```cpp
void swap(basic_string& other) noexcept(
    std::allocator_traits<Allocator>::is_always_equal::value ||
    std::allocator_traits<Allocator>::propagate_on_container_swap::value
)
```

---

## 7. Exception Safety Test Milestones

| Milestone | Test Focus |
|-----------|------------|
| M2 (Lifecycle) | Copy/move noexcept |
| M4 (Capacity) | reserve/resize fail recovery |
| M5 (Append) | append fail recovery |
| M6 (Mutation) | insert/replace fail recovery |
| M9 (Allocator) | Full failing allocator tests |

---

## 8. Debug Invariant Check

In debug builds, after every mutating operation:

```cpp
#if AMSTRING_CHECK_INVARIANTS
#define AMSTRING_INVARIANT_CHECK(s) check_invariants(s)
#else
#define AMSTRING_INVARIANT_CHECK(s) ((void)0)
#endif
```

Invariant check helps catch exception safety bugs early.