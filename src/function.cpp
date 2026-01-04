//
// Created by richard on 9/24/25.
//

#include "function.h"
#include "c_api.h"
#include "container/array.h"
#include "registry.h"
#include "tensor.h"

#include <unordered_map>

namespace aethermind {

class GlobalFunctionTable {
public:
    class Entry {
    public:
        Function func_;
        String filename_;
        uint32_t lineno_;

        Entry(Function func, String filename, uint32_t lineno)
            : func_(std::move(func)), filename_(std::move(filename)), lineno_(lineno) {}
    };

    static GlobalFunctionTable* Global() {
        static GlobalFunctionTable instance;
        return &instance;
    }

    void Register(const String& name, const Function& func, bool allow_override,
                  const String& filename, uint32_t lineno) {
        if (table_.contains(name)) {
            if (!allow_override) {
                AETHERMIND_THROW(RuntimeError) << "Global Function `" << name << "` is already registered";
            }
        }
        auto* entry = new Entry(func, filename, lineno);
        table_[name] = entry;
    }

    const Entry* Get(const String& name) {
        const auto it = table_.find(name);
        if (it == table_.end()) {
            return nullptr;
        }
        return it->second;
    }

    bool Remove(const String& name) {
        const auto it = table_.find(name);
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

    String GetRegisteredLocation(const String& name) {
        if (!table_.contains(name)) {
            AETHERMIND_THROW(RuntimeError) << "Global Function `" << name << "` is not registered";
        }
        return std::format("Global function `{}` is registered at {}:{}",
                           name.c_str(), table_[name]->filename_.c_str(), table_[name]->lineno_);
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

const QualifiedName& Function::GetQualifiedName() const noexcept {
    return pimpl_->GetQualifiedName();
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

std::optional<Function> Function::GetGlobalFunction(const String& name) {
    if (const auto* entry = GlobalFunctionTable::Global()->Get(name); entry != nullptr) {
        return entry->func_;
    }
    return std::nullopt;
}

Function Function::GetGlobalFunctionRequired(const String& name) {
    auto opt_func = GetGlobalFunction(name);
    if (!opt_func.has_value()) {
        AETHERMIND_THROW(ValueError) << "Function `" << name << "` is not registered";
    }
    return opt_func.value();
}

Array<String> Function::ListGlobalFunctionNames() {
    return GlobalFunctionTable::Global()->ListNames();
}

void Registry::RegisterFunc(const String& name, const Function& func, bool allow_override,
                            const String& filename, uint32_t lineno) {
    GlobalFunctionTable::Global()->Register(name, func, allow_override, filename, lineno);
}

String Registry::GetRegisteredLocation(const String& name) {
    return GlobalFunctionTable::Global()->GetRegisteredLocation(name);
}

}// namespace aethermind

DEFINE_STATIC_FUNCTION() {
    aethermind::Registry()
            .def("ListGlobalFunctionNamesFunctor", [] {
                     auto names = aethermind::GlobalFunctionTable::Global()->ListNames();
                     auto functor = [names](int64_t i) -> aethermind::Any {
                         if (i < 0) {
                             return names.size();
                         }
                         return names[i];
                     };
                     return aethermind::Function(functor); }, __FILE__, __LINE__)
            .def("RemoveGlobalFunction", [](const aethermind::String& name) { return aethermind::GlobalFunctionTable::Global()->Remove(name); }, __FILE__, __LINE__);
}
