#ifndef AETHERMIND_TEST_INFERENCE_CLI_PROCESS_HELPERS_H
#define AETHERMIND_TEST_INFERENCE_CLI_PROCESS_HELPERS_H

#include "aethermind/base/status.h"
#include "model/test_utils.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace aethermind::test::cli_process {

struct Result {
    int exit_code = -1;
    std::string standard_output;
    std::string standard_error;
};

#if defined(__linux__)
namespace detail {

struct SpawnActions {
    posix_spawn_file_actions_t value{};
    bool initialized = false;

    SpawnActions() = default;
    SpawnActions(const SpawnActions&) = delete;
    SpawnActions& operator=(const SpawnActions&) = delete;
    ~SpawnActions() {
        if (initialized) {
            posix_spawn_file_actions_destroy(&value);
        }
    }
};

struct ChildProcess {
    pid_t pid = -1;

    ChildProcess() = default;
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    ~ChildProcess() {
        if (pid > 0) {
            kill(pid, SIGKILL);
            int status = 0;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        }
    }
};

inline Status SystemError(const char* operation, int error) {
    return Status::Internal(std::string(operation) + ": " + std::strerror(error));
}

inline StatusOr<std::string> ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return Status::NotFound("Cannot read CLI capture: " + path.string());
    }
    std::string contents{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        return Status::Internal("Cannot finish reading CLI capture: " + path.string());
    }
    return contents;
}

} // namespace detail
#endif

/// @brief Executes a CLI without a shell and separately captures its two streams.
/// @note Temporary files avoid pipe backpressure; the child is always reaped,
/// including on timeout or an exception while the parent owns it.
inline StatusOr<Result> Run(const std::filesystem::path& executable,
                            const std::vector<std::string>& arguments) {
#if defined(__linux__)
    TempDirectory directory;
    const auto stdout_path = directory.path() / "stdout.txt";
    const auto stderr_path = directory.path() / "stderr.txt";

    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.push_back(executable.string());
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& value: storage) {
        argv.push_back(value.data());
    }
    argv.push_back(nullptr);

    detail::SpawnActions actions;
    int error = posix_spawn_file_actions_init(&actions.value);
    if (error != 0) {
        return detail::SystemError("posix_spawn_file_actions_init", error);
    }
    actions.initialized = true;
    error = posix_spawn_file_actions_addopen(&actions.value, STDOUT_FILENO,
                                             stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (error != 0) {
        return detail::SystemError("redirect CLI stdout", error);
    }
    error = posix_spawn_file_actions_addopen(&actions.value, STDERR_FILENO,
                                             stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (error != 0) {
        return detail::SystemError("redirect CLI stderr", error);
    }

    detail::ChildProcess child;
    error = posix_spawn(&child.pid, executable.c_str(), &actions.value, nullptr,
                        argv.data(), ::environ);
    if (error != 0) {
        child.pid = -1;
        return detail::SystemError("posix_spawn", error);
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    int status = 0;
    for (;;) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::DeadlineExceeded("CLI did not finish within 30 seconds");
        }
        const pid_t waited = waitpid(child.pid, &status, WNOHANG);
        if (waited == child.pid) {
            child.pid = -1;
            break;
        }
        if (waited < 0) {
            const int wait_error = errno;
            if (wait_error == EINTR) {
                continue;
            }
            if (wait_error == ECHILD) {
                // Another reaper no longer leaves us ownership of this PID.
                child.pid = -1;
            }
            return detail::SystemError("waitpid", wait_error);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    AM_ASSIGN_OR_RETURN(auto standard_output, detail::ReadFile(stdout_path));
    AM_ASSIGN_OR_RETURN(auto standard_error, detail::ReadFile(stderr_path));
    if (!WIFEXITED(status)) {
        return Status::Aborted("CLI terminated without a normal exit; wait status=" +
                               std::to_string(status) + "; stderr: " + standard_error);
    }
    return Result{WEXITSTATUS(status), std::move(standard_output), std::move(standard_error)};
#else
    UNUSED(executable);
    UNUSED(arguments);
    return Status::Unimplemented("The CLI process integration gate requires Linux");
#endif
}

} // namespace aethermind::test::cli_process

#endif
