# amstring Storage Layout Design

## 1. Object Size Target

Target: **24 bytes** on 64-bit platforms.

```cpp
sizeof(aethermind::basic_string<char>) == 24
```

This matches fbstring/folly and allows optimal SSO capacity.

---

## 2. Storage Categories

### 2.1 Three-State Model

| Category | Description | Ownership |
|----------|-------------|-----------|
| **Small** | Inline buffer (SSO) | Stack allocated, no heap |
| **Medium** | Heap allocated | Exclusive ownership |
| **Large** | Heap allocated | Reserved, same as Medium for first version |

First version: **Small + Heap exclusive**. Large category reserved but no special strategy.

### 2.2 Category Encoding

First version uses **CharT-level encoding** (not byte-level trick):

```
Small:    Last CharT stores encoded metadata = kSmallCapacity - size
Medium:   capacity_with_category stores category in bits (endian-aware)
Large:    capacity_with_category stores category in bits (endian-aware)
```

This is simpler than fbstring byte-level marker and easier to support multiple CharT types.

**Important**: Heap category encoding is **endian-aware**:

| Endian | Category Location | categoryExtractMask |
|--------|-------------------|---------------------|
| Little | `cap_` high byte (byte 7, top 2 bits) | `0xC0` |
| Big | `cap_` low byte (byte 0, low 2 bits) | `0x03` |

---

## 3. Memory Layout

### 3.1 MediumLarge Structure

```cpp
template <class CharT>
struct MediumLarge {
    CharT* data_;       // 8 bytes (pointer)
    size_t size_;       // 8 bytes
    size_t cap_;        // 8 bytes (capacity with category bits)
};
// Total: 24 bytes
```

### 3.2 Small Storage

```cpp
template <class CharT>
union Storage {
    CharT small_[kSmallArraySize];     // Inline buffer
    MediumLarge<CharT> ml_;            // Heap metadata
    std::byte bytes_[24];              // Raw bytes
};

constexpr size_t kSmallArraySize = sizeof(MediumLarge<CharT>) / sizeof(CharT);
constexpr size_t kSmallCapacity = kSmallArraySize - 1;  // Last CharT for metadata
```

### 3.3 Small Capacity by CharT

| CharT | sizeof(CharT) | Small Array Size | Small Capacity |
|-------|---------------|------------------|----------------|
| `char` | 1 | 24 | 23 |
| `char8_t` | 1 | 24 | 23 |
| `char16_t` | 2 | 12 | 11 |
| `char32_t` | 4 | 6 | 5 |
| `wchar_t` (Linux) | 4 | 6 | 5 |
| `wchar_t` (Windows) | 2 | 12 | 11 |

---

## 4. Small Metadata Encoding

### 4.1 Encoding Scheme

```
small_[kSmallCapacity] = (kSmallCapacity - size) << shift
small_[size] = CharT{}  // null terminator
```

Where `shift` depends on endian (0 for little-endian, 2 for big-endian).

### 4.2 Size Calculation

```cpp
size_type small_size() const {
    size_type diff = static_cast<size_type>(small_[kSmallCapacity]) >> shift;
    return kSmallCapacity - diff;
}
```

---

## 5. Heap Capacity Encoding（Endian-Aware）

### 5.1 Endian Constants

```cpp
constexpr bool kIsLittleEndian = std::endian::native == std::endian::little;
constexpr bool kIsBigEndian = std::endian::native == std::endian::big;
constexpr size_t kCategoryShift = (sizeof(size_t) - 1) * 8;  // 56 on 64-bit

// Category extraction mask (endian-aware)
constexpr uint8_t kCategoryExtractMask = kIsLittleEndian ? 0xC0 : 0x03;

// Capacity extraction mask (endian-aware)
constexpr size_t kCapacityExtractMask = kIsLittleEndian 
    ? ~(static_cast<size_t>(kCategoryExtractMask) << kCategoryShift)  // 0x3FFFFFFFFFFFFFF
    : 0x0;  // Unused on big-endian (use shift instead)
```

### 5.2 Category Encoding in cap_

**Little-Endian**: Category in high byte (byte 7, top 2 bits)
```cpp
cap_ = actual_capacity | (static_cast<size_t>(category) << kCategoryShift)
// Example: cap = 100, category = Medium (1)
// cap_ = 100 | (1 << 56) = 100 | 0x0100000000000000
```

**Big-Endian**: Category in low byte (byte 0, low 2 bits)
```cpp
cap_ = (actual_capacity << 2) | static_cast<size_t>(category)
// Example: cap = 100, category = Medium (1)
// cap_ = (100 << 2) | 1 = 400 | 1 = 401
```

### 5.3 Capacity Extraction（Endian-Aware）

```cpp
size_type capacity() const noexcept {
    if (is_small()) return kSmallCapacity;
    
    if (kIsLittleEndian) {
        return ml_.cap_ & kCapacityExtractMask;
    } else {
        return ml_.cap_ >> 2;
    }
}
```

### 5.4 Category Extraction

```cpp
Category category() const noexcept {
    if (is_small()) return Category::Small;
    
    if (kIsLittleEndian) {
        return static_cast<Category>((ml_.cap_ >> kCategoryShift) & 0x03);
    } else {
        return static_cast<Category>(ml_.cap_ & 0x03);
    }
}
```

### 5.5 Set Capacity with Category（Endian-Aware）

```cpp
void set_capacity(size_type cap, Category cat) noexcept {
    if (kIsLittleEndian) {
        ml_.cap_ = cap | (static_cast<size_t>(cat) << kCategoryShift);
    } else {
        ml_.cap_ = (cap << 2) | static_cast<size_t>(cat);
    }
}
```

---

## 6. Null Terminator Guarantee

All states must maintain:
```
data()[size()] == CharT{}
```

This ensures `c_str()` is always valid.

---

## 7. First Version Design Decisions (Frozen)

| Decision | Value |
|----------|-------|
| Object size | 24 bytes target |
| Small capacity | `sizeof(MediumLarge)/sizeof(CharT) - 1` |
| Small encoding | CharT-level (no byte trick), shift=0/2 for endian |
| Heap category | Endian-aware encoding in cap_ |
| Large strategy | Reserved, same as Medium |
| COW | Disabled |
| safe over-read | Disabled |
| SIMD | Disabled |

---

## 8. Future Optimizations (Deferred)

These optimizations are planned for Milestone 13 (char-optimized core):

- **safe over-read**: Block copy 24 bytes if within single 4KB page
- **branchless size()**: CMOV trick for size computation
- **byte-level marker**: Like fbstring, for char only
- **SIMD find**: memchr/memmem optimization

Do NOT implement these in first version (Milestone 1-11).