# amstring TDD Plan

## 1. Testing Philosophy

**Core principle**: Write tests first, then implement.

Every feature follows this cycle:
1. Write unit test (define expected behavior)
2. Write differential test (compare with std::string)
3. Write boundary test
4. Write invariant test
5. Write exception test (if applicable)
6. Implement minimal code
7. Run unit tests
8. Run ASan/UBSan/LSan
9. Run random differential tests
10. Stabilize before benchmark

---

## 2. Test Categories

### 2.1 Unit Tests

Location: `tests/unit/amstring/`

Purpose: Verify individual API behavior.

| Test File | Purpose | Milestone |
|-----------|---------|-----------|
| `test_core_layout.cpp` | Storage layout, category, capacity | M1 |
| `test_core_lifecycle.cpp` | Constructor, destructor, copy, move | M2 |
| `test_api_basic.cpp` | Public API basics | M3 |
| `test_api_capacity.cpp` | reserve, resize, clear | M4 |
| `test_api_append.cpp` | push_back, pop_back, append | M5 |
| `test_api_mutation.cpp` | assign, erase, insert, replace | M6 |
| `test_api_find_compare.cpp` | find, compare, starts_with | M7 |
| `test_multi_char.cpp` | char8_t, char16_t, char32_t | M8 |
| `test_allocator.cpp` | Allocator behavior | M9 |

### 2.2 Differential Tests

Compare behavior with std::basic_string.

Key checks:
- `size()` matches
- `data()` content matches
- null terminator correct
- `find()` result matches
- `compare()` result matches
- exception behavior matches

### 2.3 Invariant Tests

Every mutating operation must preserve:

```cpp
void check_invariants(const basic_string& s) {
    EXPECT_NE(s.data(), nullptr);
    EXPECT_EQ(s.data()[s.size()], CharT{});
    EXPECT_LE(s.size(), s.capacity());
}
```

### 2.4 Random Tests

Generate random operation sequences, compare with std::string.

Operations:
- construct, assign, clear, reserve, resize
- push_back, pop_back, append
- erase, insert, replace
- find, compare, shrink_to_fit

---

## 3. Invariant Definitions

| Invariant | Description |
|-----------|-------------|
| `data() != nullptr` | Always valid pointer |
| `data()[size()] == CharT{}` | Null terminator at end |
| `size() <= capacity()` | Size within capacity |
| `begin() == data()` | Iterator consistency |
| `end() == data() + size()` | Iterator consistency |
| moved-from valid | Empty string, not dangling |
| category O(1) | Small/heap distinguishable instantly |
| heap valid | data points to allocated memory |
| capacity clean | No category bits in returned capacity |

---

## 4. Test Patterns

### 4.1 Layout Test Pattern

```cpp
TEST(CoreLayout, SmallEmpty) {
    aethermind::detail::basic_string_core<char> core;
    EXPECT_EQ(core.size(), 0);
    EXPECT_EQ(core.capacity(), kSmallCapacity);
    EXPECT_EQ(core.category(), Category::Small);
    EXPECT_EQ(core.data()[0], '\0');
}
```

### 4.2 Differential Test Pattern

```cpp
TEST(Differential, ConstructFromLiteral) {
    StringPair<char> pair;
    pair.am_str = aethermind::string("hello");
    pair.std_str = std::string("hello");
    pair.check_equal();
}
```

### 4.3 Self-Reference Test Pattern

```cpp
TEST(SelfReference, AppendSelf) {
    aethermind::string s = "hello";
    s.append(s);  // Must not cause use-after-free
    EXPECT_EQ(s, "hellohello");
}
```

---

## 5. Sanitizer Testing

All tests must pass with:
- **ASan**: Address sanitizer (memory errors)
- **UBSan**: Undefined behavior sanitizer
- **LSan**: Leak sanitizer (memory leaks)

### Sanitizer Build

```bash
cmake -S . -B build-san -DENABLE_ASAN=ON -DENABLE_UBSAN=ON
cmake --build build-san --target aethermind_unit_tests
./build-san/tests/unit/aethermind_unit_tests --gtest_filter=Amstring.*
```

---

## 6. Random Test Framework

```cpp
template<typename CharT>
class RandomStringTest {
    std::mt19937 rng_;
    StringPair<CharT> pair_;
    
    void random_operation() {
        int op = rng_() % kOperationCount;
        switch (op) {
            case 0: random_construct(); break;
            case 1: random_append(); break;
            case 2: random_erase(); break;
            // ...
        }
        pair_.check_equal();
        check_invariants(pair_.am_str);
    }
};
```

---

## 7. Test Coverage Goals

| Milestone | Test Count Target |
|-----------|-------------------|
| M1 (Layout) | 10+ tests |
| M2 (Lifecycle) | 15+ tests |
| M3 (Basic API) | 20+ tests |
| M4 (Capacity) | 15+ tests |
| M5 (Append) | 20+ tests |
| M6 (Mutation) | 30+ tests |
| M7 (Find/Compare) | 20+ tests |
| M8 (Multi CharT) | 5+ tests per CharT |
| M9 (Allocator) | 15+ tests |

---

## 8. Fuzz Testing (Milestone 11)

Use libFuzzer to find edge cases.

```cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Parse data as operation sequence
    // Apply to both aethermind::string and std::string
    // Check consistency
    return 0;
}
```

---

## 9. Regression Test Policy

Every bug found becomes a regression test:
- Fuzzer crash → regression test
- Differential mismatch → regression test
- Sanitizer error → regression test

Regression tests live in `tests/unit/amstring/test_regression.cpp`.