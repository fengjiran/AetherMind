#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

class RegistryTestOperator final : public Operator {
public:
    using Params = RmsNormParams;

    explicit RegistryTestOperator(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kSiluMul;
    }

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override {
        if (ctx.backend == nullptr) {
            return Status::InvalidArgument("RegistryTestOperator backend is null");
        }
        prepared_ = true;
        return Status::Ok();
    }

    AM_NODISCARD Status Run(KernelContext&,
                            const RuntimeBindingContext&,
                            size_t) const noexcept override {
        return prepared_ ? Status::Ok() : Status::FailedPrecondition("not prepared");
    }

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_;
    }

private:
    Params params_{};
    bool prepared_ = false;
    ResolvedKernel resolved_{};
};

}// namespace

TEST(OperatorRegistry, CreateFromStaticallyRegisteredOperator) {
    // SiluMulOp is registered via AM_REGISTER_OPERATOR in silu_mul_op.cpp.
    const StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(
            OpType::kSiluMul,
            OpParams{SiluMulParams{}});

    ASSERT_TRUE(op.ok()) << op.status().ToString();
    ASSERT_NE(op.value(), nullptr);
    EXPECT_EQ(op.value()->Type(), OpType::kSiluMul);
    EXPECT_STREQ(op.value()->Name(), ToString(OpType::kSiluMul));
}

TEST(OperatorRegistry, RejectsDuplicateFactory) {
    const Status first = OperatorRegistry::Register(
            OpType::kKVCacheUpdate,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
            });
    ASSERT_TRUE(first.ok()) << first.ToString();

    const Status duplicate = OperatorRegistry::Register(
            OpType::kKVCacheUpdate,
            OperatorRegistry::Descriptor{
                    .factory_ = [](const OpParams&) -> StatusOr<std::unique_ptr<Operator>> {
                        return Status::Internal("duplicate factory should not be used");
                    },
            });

    EXPECT_EQ(duplicate.code(), StatusCode::kAlreadyExists);
}

TEST(OperatorRegistry, RegisterRejectsInvalidArguments) {
    const Status unknown = OperatorRegistry::Register(
            OpType::kUnknown,
            OperatorRegistry::Descriptor{
                    .factory_ = [](const OpParams&) -> StatusOr<std::unique_ptr<Operator>> {
                        return Status::Internal("unknown op factory should not be used");
                    },
            });
    EXPECT_EQ(unknown.code(), StatusCode::kInvalidArgument);

    const Status missing_factory = OperatorRegistry::Register(OpType::kSoftmax, OperatorRegistry::Descriptor{});
    EXPECT_EQ(missing_factory.code(), StatusCode::kInvalidArgument);
}

TEST(OperatorRegistry, CreateDefaultParamsReturnsRegisteredDefaults) {
    const Status registered = OperatorRegistry::Register(
            OpType::kRoPE,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
                    .make_default_params_ = []() -> StatusOr<OpParams> {
                        return OpParams{RegistryTestOperator::Params{.eps = 123.0F}};
                    },
            });
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kRoPE);

    ASSERT_TRUE(params.ok()) << params.status().ToString();
    const auto* typed_params = std::get_if<RegistryTestOperator::Params>(&params.value());
    ASSERT_NE(typed_params, nullptr);
    EXPECT_FLOAT_EQ(typed_params->eps, 123.0F);
}

TEST(OperatorRegistry, CreateDefaultParamsReturnsMatMulDefaults) {
    // MatMulOp is registered via AM_REGISTER_OPERATOR in matmul_op.cpp, which
    // installs both factory_ and make_default_params_. The default params
    // must be MatMulParams{transpose_rhs = false}.
    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kMatMul);
    ASSERT_TRUE(params.ok()) << params.status().ToString();
    const auto* typed_params = std::get_if<MatMulParams>(&params.value());
    ASSERT_NE(typed_params, nullptr);
    EXPECT_FALSE(typed_params->transpose_rhs);
}

TEST(OperatorRegistry, CreateDefaultParamsFailsForUnregisteredOperator) {
    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kArgmax);

    ASSERT_FALSE(params.ok());
    EXPECT_EQ(params.status().code(), StatusCode::kNotFound);
}

TEST(OperatorRegistry, CreateMissingFactoryFails) {
    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kArgmax, OpParams{});

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kNotFound);
}

TEST(OperatorRegistry, CreateUnknownOperatorFails) {
    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kUnknown, OpParams{});

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorRegistry, WrongParamsTypeFails) {
    const Status registered = OperatorRegistry::Register(
            OpType::kSoftmax,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
            });
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kSoftmax, OpParams{ArgmaxParams{}});

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kInvalidArgument);
}
