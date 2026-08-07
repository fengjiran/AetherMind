#ifndef AETHERMIND_LRU_H
#define AETHERMIND_LRU_H

/// @file
/// @brief Fixed-capacity least-recently-used cache.

#include <list>
#include <mutex>
#include <unordered_map>

namespace aethermind {

/// @brief Stores a bounded set of key-value pairs ordered by recent access.
///
/// Cache hits move the corresponding entry to the front of the recency list.
/// When the capacity is full, insertion evicts the least recently used entry.
/// Lookup and insertion are average O(1) operations.
///
/// @tparam Key Hashable key type with equality comparison.
/// @tparam Value Value type that is copy-constructible and assignable.
/// @note This class is not thread-safe. Synchronize access externally when a
///       cache is shared between threads.
template<typename Key, typename Value>
class LRUCache {
public:
    /// @brief Creates an empty cache with a fixed capacity.
    /// @param cap Maximum number of entries. Must be greater than zero.
    explicit LRUCache(size_t cap) : cap_(cap) {}

    /// @brief Looks up a key and marks a hit as the most recently used entry.
    /// @param key Key to look up.
    /// @param res Destination for the copied value when the key is present.
    /// @return True if the key was found; false otherwise. On a miss, `res` is
    ///         unchanged.
    bool get(const Key& key, Value& res) {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return false;
        }

        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        res = it->second->second;
        return true;
    }

    /// @brief Inserts or updates a value and marks it as most recently used.
    /// @param key Key to insert or update.
    /// @param value Value to copy into the cache.
    /// @note Updating an existing key does not change the cache size. Inserting
    ///       into a full cache evicts its least recently used entry.
    void put(const Key& key, const Value& value) {
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            it->second->second = value;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }

        if (cache_list_.size() == cap_) {
            const auto& last_pair = cache_list_.back();
            cache_map_.erase(last_pair.first);
            cache_list_.pop_back();
        }

        cache_list_.push_front(key, value);
        cache_map_[key] = cache_list_.begin();
    }

private:
    size_t cap_;
    std::list<std::pair<Key, Value>> cache_list_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> cache_map_;
};

}// namespace aethermind

#endif// AETHERMIND_LRU_H
