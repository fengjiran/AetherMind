#ifndef AETHERMIND_BACKEND_BACKEND_FWD_H
#define AETHERMIND_BACKEND_BACKEND_FWD_H

#include <cstdint>

namespace aethermind {

class Backend;
class BackendFactory;
class BackendRegistry;
class KernelRegistry;

enum class OpType : uint16_t;
enum class IsaLevel : uint8_t;
enum class ExecPhase : uint8_t;
enum class WeightFormat : uint8_t;

struct BackendCapabilities;
struct KernelDescriptor;
struct KernelInvocation;
struct KernelSelector;
struct ResolvedKernel;

}// namespace aethermind
#endif
