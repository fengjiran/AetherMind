#include "aethermind/operators/operator_registry.h"

#include "aethermind/backend/kernel_context.h"
#include "aethermind/runtime/workspace.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

class RegistryTestOperator final : public Operator {
public:
    struct Params {
        int value_ = 0;
    };

    explicit RegistryTestOperator(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kAdd;
    }

    AM_NODISCARD Status ValidateParams() const override {
        if (params_.value_ <= 0) {
            return Status::InvalidArgument("RegistryTestOperator value must be positive");
        }
        return Status::Ok();
    }

    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override {
        if (!inputs.empty()) {
            return Status::InvalidArgument("RegistryTestOperator expects no inputs");
        }
        return Status::Ok();
    }

    AM_NODISCARD StatusOr<std::vector<TensorSpec>> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override {
        if (!inputs.empty()) {
            return Status::InvalidArgument("RegistryTestOperator expects no shape inputs");
        }
        return std::vector<TensorSpec>{};
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
        return prepared_ ? Status::Ok() : Status(StatusCode::kFailedPrecondition, "not prepared");
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
            &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>);
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(
            OpType::kAdd,
            RegistryTestOperator::Params{.value_ = 7});

    ASSERT_TRUE(op.ok());
    ASSERT_NE(op.value(), nullptr);
    EXPECT_EQ(op.value()->Type(), OpType::kAdd);
    EXPECT_STREQ(op.value()->Name(), ToString(OpType::kAdd));
    EXPECT_TRUE(op.value()->ValidateParams().ok());
}

TEST(OperatorRegistry, RejectsDuplicateFactory) {
    const Status first = OperatorRegistry::Register(
            OpType::kElementwiseMul,
            &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>);
    ASSERT_TRUE(first.ok()) << first.ToString();

    const Status duplicate = OperatorRegistry::Register(
            OpType::kElementwiseMul,
            [](const std::any&) -> StatusOr<std::unique_ptr<Operator>> {
                return Status::Internal("duplicate factory should not be used");
            });

    EXPECT_EQ(duplicate.code(), StatusCode::kAlreadyExists);
}

TEST(OperatorRegistry, RegisterRejectsInvalidArguments) {
    const Status unknown = OperatorRegistry::Register(
            OpType::kUnknown,
            [](const std::any&) -> StatusOr<std::unique_ptr<Operator>> {
                return Status::Internal("unknown op factory should not be used");
            });
    EXPECT_EQ(unknown.code(), StatusCode::kInvalidArgument);

    const Status missing_factory = OperatorRegistry::Register(OpType::kSoftmax, OperatorRegistry::FactoryFunc{});
    EXPECT_EQ(missing_factory.code(), StatusCode::kInvalidArgument);
}

TEST(OperatorRegistry, CreateDefaultParamsReturnsRegisteredDefaults) {
    const Status registered = OperatorRegistry::Register(
            OpType::kLinear,
            OperatorRegistry::Descriptor{
                    .factory_ = &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>,
                    .make_default_params_ = []() -> StatusOr<std::any> {
                        return std::any{RegistryTestOperator::Params{.value_ = 123}};
                    },
            });
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    const StatusOr<std::any> params = OperatorRegistry::CreateDefaultParams(OpType::kLinear);

    ASSERT_TRUE(params.ok()) << params.status().ToString();
    const auto* typed_params = std::any_cast<RegistryTestOperator::Params>(&params.value());
    ASSERT_NE(typed_params, nullptr);
    EXPECT_EQ(typed_params->value_, 123);
}

TEST(OperatorRegistry, CreateDefaultParamsFailsForFactoryOnlyRegistration) {
    const Status registered = OperatorRegistry::Register(
            OpType::kMatMul,
            &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>);
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    const StatusOr<std::any> params = OperatorRegistry::CreateDefaultParams(OpType::kMatMul);

    ASSERT_FALSE(params.ok());
    EXPECT_EQ(params.status().code(), StatusCode::kNotFound);
}

TEST(OperatorRegistry, CreateDefaultParamsFailsForUnregisteredOperator) {
    const StatusOr<std::any> params = OperatorRegistry::CreateDefaultParams(OpType::kArgmax);

    ASSERT_FALSE(params.ok());
    EXPECT_EQ(params.status().code(), StatusCode::kNotFound);
}

TEST(OperatorRegistry, CreateMissingFactoryFails) {
    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kArgmax, std::any{});

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kNotFound);
}

TEST(OperatorRegistry, CreateUnknownOperatorFails) {
    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kUnknown, std::any{});

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kInvalidArgument);
}

TEST(OperatorRegistry, WrongParamsTypeFails) {
    const Status registered = OperatorRegistry::Register(
            OpType::kSilu,
            &OperatorRegistry::CreateTypedOperator<RegistryTestOperator>);
    ASSERT_TRUE(registered.ok()) << registered.ToString();

    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kSilu, 7);

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kInvalidArgument);
}
