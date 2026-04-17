#ifndef AETHERMIND_BACKEND_STREAM_H
#define AETHERMIND_BACKEND_STREAM_H

namespace aethermind {

class Stream {
public:
    virtual ~Stream() = default;
};

class CpuInlineStream final : public Stream {
public:
    ~CpuInlineStream() override = default;
};

}// namespace aethermind

#endif
