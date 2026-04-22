# amstring API Compatibility Matrix

This document tracks std::basic_string API coverage in aethermind::basic_string.

---

## 1. API Coverage Overview

| Status | Meaning |
|--------|---------|
| ✅ | Implemented |
| ⏳ | Planned (milestone noted) |
| ❌ | Not planned for MVP |
| 🔜 | Future version |

---

## 2. Constructors

### 2.1 Basic Constructors

| API | Status | Milestone |
|-----|--------|-----------|
| `basic_string()` | ⏳ | M3 |
| `basic_string(const basic_string&)` | ⏳ | M3 |
| `basic_string(basic_string&&)` | ⏳ | M3 |
| `basic_string(const CharT*, size_type)` | ⏳ | M3 |
| `basic_string(const CharT*)` | ⏳ | M3 |
| `basic_string(size_type, CharT)` | ⏳ | M3 |
| `basic_string(const basic_string&, size_type, size_type)` | 🔜 | Post-MVP |
| `basic_string(InputIt, InputIt)` | 🔜 | Post-MVP |
| `basic_string(std::initializer_list<CharT>)` | 🔜 | Post-MVP |
| `basic_string(std::basic_string_view<CharT>)` | ⏳ | M3 |

### 2.2 Allocator Constructors

| API | Status | Milestone |
|-----|--------|-----------|
| `basic_string(const Allocator&)` | ⏳ | M9 |
| `basic_string(const basic_string&, const Allocator&)` | ⏳ | M9 |
| `basic_string(basic_string&&, const Allocator&)` | ⏳ | M9 |

---

## 3. Assignment

| API | Status | Milestone |
|-----|--------|-----------|
| `operator=(const basic_string&)` | ⏳ | M3 |
| `operator=(basic_string&&)` | ⏳ | M3 |
| `operator=(const CharT*)` | ⏳ | M3 |
| `operator=(CharT)` | ⏳ | M3 |
| `assign(const basic_string&)` | ⏳ | M6 |
| `assign(const basic_string&, size_type, size_type)` | 🔜 | Post-MVP |
| `assign(const CharT*, size_type)` | ⏳ | M6 |
| `assign(const CharT*)` | ⏳ | M6 |
| `assign(size_type, CharT)` | ⏳ | M6 |
| `assign(InputIt, InputIt)` | 🔜 | Post-MVP |
| `assign(std::initializer_list<CharT>)` | 🔜 | Post-MVP |

---

## 4. Element Access

| API | Status | Milestone |
|-----|--------|-----------|
| `operator[](size_type)` | ⏳ | M3 |
| `operator[](size_type) const` | ⏳ | M3 |
| `at(size_type)` | ⏳ | M3 |
| `at(size_type) const` | ⏳ | M3 |
| `front()` | ⏳ | M3 |
| `front() const` | ⏳ | M3 |
| `back()` | ⏳ | M3 |
| `back() const` | ⏳ | M3 |
| `data()` | ⏳ | M3 |
| `data() const` | ⏳ | M3 |
| `c_str() const` | ⏳ | M3 |
| `operator std::basic_string_view<CharT>() const` | 🔜 | Post-MVP |

---

## 5. Iterators

| API | Status | Milestone |
|-----|--------|-----------|
| `begin() / cbegin()` | ⏳ | M3 |
| `end() / cend()` | ⏳ | M3 |
| `rbegin() / crbegin()` | 🔜 | Post-MVP |
| `rend() / crend()` | 🔜 | Post-MVP |

---

## 6. Capacity

| API | Status | Milestone |
|-----|--------|-----------|
| `size() const` | ⏳ | M1 |
| `length() const` | ⏳ | M3 |
| `max_size() const` | ⏳ | M4 |
| `resize(size_type)` | ⏳ | M4 |
| `resize(size_type, CharT)` | ⏳ | M4 |
| `capacity() const` | ⏳ | M1 |
| `reserve(size_type)` | ⏳ | M4 |
| `clear()` | ⏳ | M4 |
| `empty() const` | ⏳ | M1 |
| `shrink_to_fit()` | ⏳ | M4 |

---

## 7. Modifiers

### 7.1 Append

| API | Status | Milestone |
|-----|--------|-----------|
| `operator+=(const basic_string&)` | ⏳ | M5 |
| `operator+=(const CharT*)` | ⏳ | M5 |
| `operator+=(CharT)` | ⏳ | M5 |
| `operator+=(std::initializer_list<CharT>)` | 🔜 | Post-MVP |
| `append(const basic_string&)` | ⏳ | M5 |
| `append(const basic_string&, size_type, size_type)` | 🔜 | Post-MVP |
| `append(const CharT*, size_type)` | ⏳ | M5 |
| `append(const CharT*)` | ⏳ | M5 |
| `append(size_type, CharT)` | ⏳ | M5 |
| `append(InputIt, InputIt)` | 🔜 | Post-MVP |
| `push_back(CharT)` | ⏳ | M5 |
| `pop_back()` | ⏳ | M5 |

### 7.2 Insert

| API | Status | Milestone |
|-----|--------|-----------|
| `insert(size_type, const basic_string&)` | ⏳ | M6 |
| `insert(size_type, const basic_string&, size_type, size_type)` | 🔜 | Post-MVP |
| `insert(size_type, const CharT*, size_type)` | ⏳ | M6 |
| `insert(size_type, const CharT*)` | ⏳ | M6 |
| `insert(size_type, size_type, CharT)` | ⏳ | M6 |
| `insert(const_iterator, CharT)` | 🔜 | Post-MVP |
| `insert(const_iterator, size_type, CharT)` | 🔜 | Post-MVP |
| `insert(const_iterator, InputIt, InputIt)` | 🔜 | Post-MVP |

### 7.3 Erase

| API | Status | Milestone |
|-----|--------|-----------|
| `erase(size_type = 0, size_type = npos)` | ⏳ | M6 |
| `erase(const_iterator)` | 🔜 | Post-MVP |
| `erase(const_iterator, const_iterator)` | 🔜 | Post-MVP |

### 7.4 Replace

| API | Status | Milestone |
|-----|--------|-----------|
| `replace(size_type, size_type, const basic_string&)` | ⏳ | M6 |
| `replace(size_type, size_type, const CharT*, size_type)` | ⏳ | M6 |
| `replace(size_type, size_type, const CharT*)` | ⏳ | M6 |
| `replace(size_type, size_type, size_type, CharT)` | ⏳ | M6 |
| `replace(const_iterator, const_iterator, ...)` | 🔜 | Post-MVP |

---

## 8. Search

| API | Status | Milestone |
|-----|--------|-----------|
| `find(const basic_string&, size_type = 0)` | ⏳ | M7 |
| `find(const CharT*, size_type, size_type)` | ⏳ | M7 |
| `find(const CharT*, size_type = 0)` | ⏳ | M7 |
| `find(CharT, size_type = 0)` | ⏳ | M7 |
| `rfind(...)` | ⏳ | M7 |
| `find_first_of(...)` | 🔜 | Post-MVP |
| `find_first_not_of(...)` | 🔜 | Post-MVP |
| `find_last_of(...)` | 🔜 | Post-MVP |
| `find_last_not_of(...)` | 🔜 | Post-MVP |
| `starts_with(...)` (C++20) | ⏳ | M7 |
| `ends_with(...)` (C++20) | ⏳ | M7 |
| `contains(...)` (C++23) | ⏳ | M7 |

---

## 9. Compare

| API | Status | Milestone |
|-----|--------|-----------|
| `compare(const basic_string&)` | ⏳ | M7 |
| `compare(size_type, size_type, const basic_string&)` | 🔜 | Post-MVP |
| `compare(const CharT*)` | ⏳ | M7 |

---

## 10. Standard Library Integration

| API | Status | Milestone |
|-----|--------|-----------|
| `operator==` | ⏳ | M10 |
| `operator!=` | ⏳ | M10 |
| `operator<` | ⏳ | M10 |
| `operator<=` | ⏳ | M10 |
| `operator>` | ⏳ | M10 |
| `operator>=` | ⏳ | M10 |
| `operator<=>` (C++20) | ⏳ | M10 |
| `operator+` | ⏳ | M10 |
| `std::hash<basic_string>` | ⏳ | M10 |
| `std::swap(basic_string)` | ⏳ | M10 |
| `operator<<` / `operator>>` | 🔜 | Post-MVP |
| `std::formatter<basic_string>` (C++23) | 🔜 | Post-MVP |

---

## 11. MVP API Scope

First version MVP (Milestones 1-7) includes:

- ✅ Basic constructors (default, copy, move, cstr, pointer-length, count-char)
- ✅ Assignment operators
- ✅ Element access (data, c_str, operator[], at, front, back)
- ✅ Iterators (begin, end)
- ✅ Capacity (size, capacity, empty, reserve, resize, clear, shrink_to_fit)
- ✅ Append (push_back, pop_back, append, operator+=)
- ✅ Mutation (assign, erase, insert, replace - basic forms)
- ✅ Search (find, rfind, starts_with, ends_with, contains)
- ✅ Compare

Not in MVP:
- ❌ Full iterator overloads
- ❌ Full replace/insert overloads
- ❌ find_first_of/last_of variants
- ❌ Allocator propagation
- ❌ Stream I/O
- ❌ Formatter