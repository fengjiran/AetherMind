#include "aethermind/inference/inference_session.h"
#include "aethermind/runtime/workspace_planner.h"

#include <span>

#ifdef _MSVC_LANG
static_assert(_MSVC_LANG >= 202002L);
#else
static_assert(__cplusplus >= 202002L);
#endif

void ConsumeAetherMindContract(std::span<aethermind::WorkspaceRequirement>) {}
