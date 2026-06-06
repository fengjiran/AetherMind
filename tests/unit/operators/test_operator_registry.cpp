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
    ASSERT_TRUE(OperatorRegistry::Register(
            OpType::kAdd,
            [](const std::any& params) -> StatusOr<std::unique_ptr<Operator>> {
                try {
                    return std::make_unique<RegistryTestOperator>(
                            std::any_cast<RegistryTestOperator::Params>(params));
                } catch (const std::bad_any_cast&) {
                    return Status::InvalidArgument("Wrong params type for RegistryTestOperator");
                }
            }));

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
    ASSERT_TRUE(OperatorRegistry::Register(
            OpType::kElementwiseMul,
            [](const std::any& params) -> StatusOr<std::unique_ptr<Operator>> {
                try {
                    return std::make_unique<RegistryTestOperator>(
                            std::any_cast<RegistryTestOperator::Params>(params));
                } catch (const std::bad_any_cast&) {
                    return Status::InvalidArgument("Wrong params type for RegistryTestOperator");
                }
            }));

    const bool registered = OperatorRegistry::Register(
            OpType::kElementwiseMul,
            [](const std::any&) -> StatusOr<std::unique_ptr<Operator>> {
                return Status::Internal("duplicate factory should not be used");
            });

    EXPECT_FALSE(registered);
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
    ASSERT_TRUE(OperatorRegistry::Register(
            OpType::kSilu,
            [](const std::any& params) -> StatusOr<std::unique_ptr<Operator>> {
                try {
                    return std::make_unique<RegistryTestOperator>(
                            std::any_cast<RegistryTestOperator::Params>(params));
                } catch (const std::bad_any_cast&) {
                    return Status::InvalidArgument("Wrong params type for RegistryTestOperator");
                }
            }));

    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(OpType::kSilu, 7);

    ASSERT_FALSE(op.ok());
    EXPECT_EQ(op.status().code(), StatusCode::kInvalidArgument);
}
