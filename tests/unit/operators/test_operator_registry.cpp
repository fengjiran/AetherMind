#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

class RegistryTestOperator final : public Operator {
public:
    using Params = RmsNormParams;

    explicit RegistryTestOperator(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kAdd;
    }

    AM_NODISCARD Status ValidateParams() const override {
        if (params_.eps <= 0.0F) {
            return Status::InvalidArgument("RegistryTestOperator eps must be positive");
        }
        return Status::Ok();
    }

    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override {
        if (!inputs.empty()) {
            return Status::InvalidArgument("RegistryTestOperator expects no inputs");
        }
        return Status::Ok();
    }

    AM_NODISCARD StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override {
        if (!inputs.empty()) {
            return Status::InvalidArgument("RegistryTestOperator expects no shape inputs");
        }
        return InferenceResult{};
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

TEST(OperatorRegistry, RegisterAndCreateOperator) {
    const Status registered = OperatorRegistry::Register(
            OpType::kAdd,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
            });
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(
            OpType::kAdd,
            OpParams{RegistryTestOperator::Params{.eps = 7.0F}});

    ASSERT_TRUE(op.ok());
    ASSERT_NE(op.value(), nullptr);
    EXPECT_EQ(op.value()->Type(), OpType::kAdd);
    EXPECT_STREQ(op.value()->Name(), ToString(OpType::kAdd));
    EXPECT_TRUE(op.value()->ValidateParams().ok());
}

TEST(OperatorRegistry, RejectsDuplicateFactory) {
    const Status first = OperatorRegistry::Register(
            OpType::kElementwiseMul,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
            });
    ASSERT_TRUE(first.ok()) << first.ToString();

    const Status duplicate = OperatorRegistry::Register(
            OpType::kElementwiseMul,
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

TEST(OperatorRegistry, CreateDefaultParamsFailsForFactoryOnlyRegistration) {
    const Status registered = OperatorRegistry::Register(
            OpType::kMatMul,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
            });
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kMatMul);

    ASSERT_FALSE(params.ok());
    EXPECT_EQ(params.status().code(), StatusCode::kNotFound);
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
            OpType::kSilu,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
            });
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kSilu, OpParams{ArgmaxParams{}});

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kInvalidArgument);
}
