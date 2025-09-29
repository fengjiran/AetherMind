//
// Created by richard on 9/24/25.
//

#include "function.h"
#include "c_api.h"
#include "registry.h"

#include <unordered_map>

namespace aethermind {

class GlobalFunctionTable {
public:
    class Entry {
    public:
        String name_{};
        String doc_{};
        String schema_{};
        Function func_{};

        Entry(String name, String doc, String schema, Function func)
            : name_(std::move(name)), doc_(std::move(doc)), schema_(std::move(schema)), func_(std::move(func)) {}
    };

    static GlobalFunctionTable* Global() {
        static GlobalFunctionTable instance;
        return &instance;
    }

    void Register(const String& name, const String& doc, const String& schema, const Function& func,
                  bool allow_override) {
        if (table_.contains(name)) {
            if (!allow_override) {
                AETHERMIND_THROW(RuntimeError) << "Global Function `" << name << "` is already registered";
            }
        }
        auto* entry = new Entry(name, doc, schema, func);
        table_[name] = entry;
    }

    const Entry* Get(const String& name) {
        auto it = table_.find(name);
        if (it == table_.end()) {
            return nullptr;
        }
        return it->second;
    }

    bool Remove(const String& name) {
        auto it = table_.find(name);
        if (it == table_.end()) {
            return false;
        }
        table_.erase(it);
        return true;
    }

    NODISCARD Array<String> ListNames() const {
        Array<String> names;
        names.reserve(table_.size());
        for (const auto& kv: table_) {
            names.push_back(kv.first);
        }
        return names;
    }

private:
    GlobalFunctionTable() = default;
    std::unordered_map<String, Entry*> table_{};
};

bool Function::defined() const noexcept {
    return pimpl_;
}

uint32_t Function::use_count() const noexcept {
    return pimpl_.use_count();
}

bool Function::unique() const noexcept {
    return pimpl_.unique();
}

const String& Function::schema() const noexcept {
    return pimpl_->schema();
}

FunctionImpl* Function::get_impl_ptr_unsafe() const noexcept {
    return pimpl_.get();
}

FunctionImpl* Function::release_impl_unsafe() noexcept {
    return pimpl_.release();
}

void Function::CallPacked(const Any* args, int32_t num_args, Any* res) const {
    pimpl_->CallPacked(args, num_args, res);
}

void Function::CallPacked(details::PackedArgs args, Any* res) const {
    pimpl_->CallPacked(args.data(), args.size(), res);
}

void Function::RegisterGlobalFunction(const String& name, const String& doc, const Function& func, bool allow_override) {
    GlobalFunctionTable::Global()->Register(name, doc, func.schema(), func, allow_override);
}

std::optional<Function> Function::GetGlobalFunction(const String& name) {
    if (const auto* entry = GlobalFunctionTable::Global()->Get(name); entry != nullptr) {
        return entry->func_;
    }
    return std::nullopt;
}

Array<String> Function::ListGlobalFunctionNames() {
    return GlobalFunctionTable::Global()->ListNames();
}

void Registry::RegisterFunc(const String& name, const String& doc, Function func, bool allow_override) {
    GlobalFunctionTable::Global()->Register(name, doc, func.schema(), func, allow_override);
}


}// namespace aethermind
