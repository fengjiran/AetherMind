//
// Created by richard on 8/16/25.
//

#ifndef AETHERMIND_TYPE_TRAITS_H
#define AETHERMIND_TYPE_TRAITS_H

#include <string>
#include <type_traits>
#include <utility>

namespace aethermind {

#define AETHERMIND_FORALL_TAGS(_) \
    _(None)                       \
    _(Tensor)                     \
    _(Storage)                    \
    _(Double)                     \
    _(ComplexDouble)              \
    _(Int)                        \
    _(SymInt)                     \
    _(SymFloat)                   \
    _(SymBool)                    \
    _(Bool)                       \
    _(Tuple)                      \
    _(String)                     \
    _(Blob)                       \
    _(GenericList)                \
    _(GenericDict)                \
    _(Future)                     \
    _(Await)                      \
    _(Device)                     \
    _(Stream)                     \
    _(Object)                     \
    _(PyObject)                   \
    _(Uninitialized)              \
    _(Capsule)                    \
    _(RRef)                       \
    _(Quantizer)                  \
    _(Generator)                  \
    _(Enum)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
    AETHERMIND_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

struct AetherMindAny {
    union Payload {
        union data {
            int64_t v_int;
            double v_double;
            bool v_bool;
            const char* v_str;
            void* v_handle;
            Device v_device;

            data() : v_int(0) {}
        };

        data u;
        Tensor v_tensor;

        static_assert(std::is_trivially_copyable_v<data>);
        Payload() : u() {}
        Payload(const Payload&) = delete;
        Payload(Payload&&) = delete;
        Payload& operator=(const Payload&) = delete;
        Payload& operator=(Payload&&) = delete;

        ~Payload() {}
    };

    Payload payload_;
    Tag tag_{Tag::None};
};


template<typename, typename = void>
struct TypeTraits;

template<>
struct TypeTraits<bool> {
    static void CopyToAny(const bool& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Bool;
        dst->payload_.u.v_bool = src;
    }

    static void MoveToAny(bool src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static bool CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return src->payload_.u.v_bool;
    }

    static bool MoveFromAnyAfterCheck(AetherMindAny* src) {
        return src->payload_.u.v_bool;
    }

    static std::optional<bool> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return src->payload_.u.v_bool;
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Bool;
    }

    static std::string TypeStr() {
        return "bool";
    }
};

// POD Int type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_integral_v<T>>> {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Int;
        dst->payload_.u.v_int = static_cast<int64_t>(src);
    }

    static void MoveToAny(T src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static T CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return static_cast<T>(src->payload_.u.v_int);
    }

    static T MoveFromAnyAfterCheck(AetherMindAny* src) {
        return static_cast<T>(src->payload_.u.v_int);
    }

    static std::optional<T> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return static_cast<T>(src->payload_.u.v_int);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Int;
    }

    static std::string TypeStr() {
        return "int";
    }
};

// POD Float type
template<typename T>
struct TypeTraits<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    static void CopyToAny(const T& src, AetherMindAny* dst) {
        dst->tag_ = Tag::Double;
        dst->payload_.u.v_double = static_cast<double>(src);
    }

    static void MoveToAny(T src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static T CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return static_cast<T>(src->payload_.u.v_double);
    }

    static T MoveFromAnyAfterCheck(AetherMindAny* src) {
        return static_cast<T>(src->payload_.u.v_double);
    }

    static std::optional<T> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return static_cast<T>(src->payload_.u.v_double);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::Double;
    }

    static std::string TypeStr() {
        return "double";
    }
};

// const char* type
template<>
struct TypeTraits<const char*> {
    static void CopyToAny(const char* src, AetherMindAny* dst) {
        CHECK(src != nullptr);
        dst->tag_ = Tag::String;
        dst->payload_.u.v_str = src;
    }

    static void MoveToAny(const char* src, AetherMindAny* dst) {
        CopyToAny(src, dst);
    }

    static std::string CopyFromAnyAfterCheck(const AetherMindAny* src) {
        return src->payload_.u.v_str;
    }

    static std::string MoveFromAnyAfterCheck(AetherMindAny* src) {
        return src->payload_.u.v_str;
    }

    static std::optional<const char*> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return src->payload_.u.v_str;
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == Tag::String;
    }

    static std::string TypeStr() {
        return "const char*";
    }
};

// string type
template<>
struct TypeTraits<std::string> {
    static void CopyToAny(const std::string& src, AetherMindAny* dst) {
        dst->tag_ = Tag::String;
        dst->payload_.u.v_str = src.c_str();
    }

    static void MoveToAny(std::string src, AetherMindAny* dst) {
        dst->tag_ = Tag::String;
        dst->payload_.u.v_str = src.c_str();
        src.clear();
    }

    static std::string TypeStr() {
        return "std::string";
    }
};

}// namespace aethermind

#endif//AETHERMIND_TYPE_TRAITS_H
