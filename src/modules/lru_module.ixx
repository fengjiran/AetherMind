//
// Created by richard on 1/28/26.
//
module;

#include <list>
#include <mutex>
#include <unordered_map>

export module LRU;

namespace aethermind {

export template<typename Key, typename Value>
class LRUCache {
public:
    explicit LRUCache(size_t cap) : cap_(cap) {}

    bool get(const Key& key, Value& res) {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return false;
        }

        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        res = it->second->second;
        return true;
    }

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